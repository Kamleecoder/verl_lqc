import asyncio
import os
from pathlib import Path

import pytest
import ray
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from tests.checkpoint_engine.test_utils import create_rollout_worker_group, create_trainer_worker_group
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AsyncLLMServerManager
from verl.single_controller.ray import RayResourcePool
from verl.single_controller.ray.base import split_resource_pool
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_name, is_torch_npu_available
from verl.workers.config import CheckpointEngineConfig, HFModelConfig

MODEL_PATH = Path(os.path.expanduser(os.environ.get("VERL_TEST_VLLM_MODEL_PATH", "~/models/Qwen/Qwen2.5-0.5B-Instruct")))

# Before running on NPU, set the model path:
#   VERL_TEST_VLLM_MODEL_PATH=~/models/Qwen/Qwen2.5-0.5B-Instruct \
#     pytest tests/workers/rollout/rollout_vllm/test_server_adapter_compare_outputs.py -v -s
#
# The test creates 2 worker groups (trainer + rollout) and requires at least 2 NPU devices.
# Each worker group gets placed on 1 device (tensor_model_parallel_size=1, data_parallel_size=1).
# Optionally specify specific NPU devices via ASCEND_RT_VISIBLE_DEVICES if needed.


def _ray_runtime_env_vars() -> dict[str, str]:
    env: dict[str, str] = {
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "true"),
        "VERL_LOGGING_LEVEL": os.environ.get("VERL_LOGGING_LEVEL", "INFO"),
        "VLLM_LOGGING_LEVEL": os.environ.get("VLLM_LOGGING_LEVEL", "INFO"),
    }
    if is_torch_npu_available():
        if "ASCEND_RT_VISIBLE_DEVICES" in os.environ:
            env["ASCEND_RT_VISIBLE_DEVICES"] = os.environ["ASCEND_RT_VISIBLE_DEVICES"]
        env.setdefault("HCCL_CONNECT_TIMEOUT", "1500")
        env.setdefault("HCCL_HOST_SOCKET_PORT_RANGE", "60000-60050")
        env.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "61000-61050")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env.setdefault("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])
    return env


def _build_base_config(model_path: str) -> DictConfig:
    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "+async_training.partial_rollout=True",
            ],
        )

    config.actor_rollout_ref.model.path = os.path.expanduser(model_path)
    config.actor_rollout_ref.rollout.name = "vllm"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
    config.actor_rollout_ref.rollout.data_parallel_size = 1
    config.actor_rollout_ref.rollout.pipeline_model_parallel_size = 1
    config.actor_rollout_ref.rollout.checkpoint_engine.backend = "nccl"
    config.actor_rollout_ref.rollout.nnodes = 1
    config.actor_rollout_ref.rollout.n_gpus_per_node = 1
    config.trainer.nnodes = 1
    config.trainer.n_gpus_per_node = 1
    return config


async def _collect_reshard_signature(rollout_wg) -> dict[str, float]:
    worker = rollout_wg.workers[0]
    return ray.get(
        worker.__ray_call__.remote(
            lambda self: {
                "num_tensors": float(len(self.server_adapter.received_weights)),
                "numel_total": float(sum(t.numel() for t in self.server_adapter.received_weights.values())),
                "l1_sum": float(sum(t.abs().double().sum().item() for t in self.server_adapter.received_weights.values())),
                "l2_sum": float(
                    sum((t.double().pow(2).sum().sqrt().item()) for t in self.server_adapter.received_weights.values())
                ),
            }
        )
    )


async def _run_once_collect_text(base_config: DictConfig, load_format: str, prompt: str) -> tuple[str, dict[str, float]]:
    config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
    config.actor_rollout_ref.rollout.load_format = load_format

    ray.init(runtime_env={"env_vars": _ray_runtime_env_vars()}, ignore_reinit_error=True)
    try:
        model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        checkpoint_engine_config: CheckpointEngineConfig = omega_conf_to_dataclass(
            config.actor_rollout_ref.rollout.checkpoint_engine
        )

        resource_pool = RayResourcePool(process_on_nodes=[2], max_colocate_count=3)
        resource_pool.get_placement_groups(device_name=get_device_name())
        trainer_pool, rollout_pool = split_resource_pool(resource_pool, [1, 1])
        trainer = create_trainer_worker_group(trainer_pool, model_config, checkpoint_engine_config)
        trainer.reset()

        # Run one real checkpoint-engine update into a rollout worker and collect
        # a numeric signature from the resharded weights.
        rollout_mock, replicas_mock = await create_rollout_worker_group(
            rollout_pool,
            model_config,
            omega_conf_to_dataclass(config.actor_rollout_ref.rollout),
            check_allclose=True,
        )
        checkpoint_manager_mock = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=trainer,
            replicas=replicas_mock,
        )
        await checkpoint_manager_mock.update_weights(global_steps=1)
        reshard_sig = await _collect_reshard_signature(rollout_mock)

        agent_loop_manager = await AgentLoopManager.create(config=config)
        servers = list(
            zip(
                agent_loop_manager.server_addresses,
                [server._server_handle for server in agent_loop_manager.rollout_replicas],
                strict=True,
            )
        )
        checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config, trainer=trainer, replicas=agent_loop_manager.rollout_replicas
        )
        server_manager = AsyncLLMServerManager(
            config=config,
            servers=servers,
            load_balancer_handle=agent_loop_manager.global_load_balancer,
        )

        prompt_ids = model_config.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
        output = await server_manager.generate(
            request_id=f"compare_server_adapter_{load_format}",
            prompt_ids=prompt_ids,
            sampling_params={"temperature": 0.0, "top_p": 1.0, "top_k": -1, "logprobs": False, "max_tokens": 96},
        )
        assert output.stop_reason in ("completed", None)
        assert len(output.token_ids) > 0
        text = model_config.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        assert text.strip() != ""
        return text, reshard_sig
    finally:
        ray.shutdown()


def test_compare_outputs_via_server_adapter():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model path does not exist: {MODEL_PATH}")

    base_config = _build_base_config(model_path=str(MODEL_PATH))
    prompt = "写一段关于昇腾的介绍"

    text_dummy, sig_dummy = asyncio.run(_run_once_collect_text(base_config, "dummy", prompt))
    text_auto, sig_auto = asyncio.run(_run_once_collect_text(base_config, "auto", prompt))

    diff_l1 = abs(sig_dummy["l1_sum"] - sig_auto["l1_sum"])
    diff_l2 = abs(sig_dummy["l2_sum"] - sig_auto["l2_sum"])

    print(
        "\n[Compare Prompt] "
        f"{prompt}\n[ReshardSig][dummy] {sig_dummy}\n[ReshardSig][auto] {sig_auto}\n"
        f"[ReshardSigDiff] l1={diff_l1:.6e}, l2={diff_l2:.6e}\n"
        f"{prompt}\n[Generated][dummy+update][ServerAdapter] {text_dummy}\n"
        f"[Generated][auto][ServerAdapter] {text_auto}\n"
    )
    assert sig_dummy["num_tensors"] == sig_auto["num_tensors"]
    assert sig_dummy["numel_total"] == sig_auto["numel_total"]
