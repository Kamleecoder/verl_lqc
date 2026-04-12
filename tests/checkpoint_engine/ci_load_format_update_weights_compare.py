# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CI helper: compare rollout behavior after update_weights(global_steps=None).

Runs two isolated setups with ``load_format`` set to ``dummy_dtensor`` and ``auto``,
each performing the same weight-update + generate path, then compares generated
``token_ids``. Exit code 1 if they differ.

**Ascend NPU (e.g. A2)** — check the following before running:

- ``ROLLOUT_NAME`` must match your inference stack (often ``vllm`` or ``sglang`` on Ascend).
- **Accelerator count**: standalone AgentLoop uses ``trainer.n_gpus_per_node`` processes for
  the trainer **and** the same count for rollout (``rollout.n_gpus_per_node`` follows Hydra
  ``${oc.select:trainer.n_gpus_per_node,...}``). You need **at least** ``2 * n_gpus_per_node``
  NPUs on the node. Override with ``--n-gpus-per-node`` or ``VERL_N_GPUS_PER_NODE``.
- **HCCL / CANN**: this script sets common HCCL timeouts and port ranges for Ray workers
  (same spirit as ``tests/checkpoint_engine/test_correctness_on_npu.py``). On CANN >= 8.5 you
  may still need ``HCCL_INTRA_ROCE_ENABLE=1`` in the shell or cluster defaults
  (see ``verl/checkpoint_engine/README.md``).
- **VLLM_* env vars** are applied only when ``ROLLOUT_NAME`` looks like a vLLM rollout
  (substring ``vllm``), so sglang-only jobs are not polluted.

Run from the repository root.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import ray
import torch
from omegaconf import DictConfig, OmegaConf

from tests.checkpoint_engine.test_utils import create_trainer_worker_group
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AsyncLLMServerManager
from verl.single_controller.ray import RayResourcePool
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import is_torch_npu_available
from verl.workers.config import CheckpointEngineConfig, HFModelConfig


def _accelerator_count() -> int:
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    if is_torch_npu_available():
        return torch.npu.device_count()
    return 0


def _ray_runtime_env_vars(rollout_name: str) -> dict[str, str]:
    """Worker env for Ray; Ascend needs HCCL-related vars, CUDA often needs NCCL/VLLM."""
    env: dict[str, str] = {
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "true"),
        "VERL_LOGGING_LEVEL": os.environ.get("VERL_LOGGING_LEVEL", "INFO"),
    }

    if is_torch_npu_available():
        # Align with tests/checkpoint_engine/test_correctness_on_npu.py (HCCL over sockets).
        env.setdefault("HCCL_CONNECT_TIMEOUT", "1500")
        env.setdefault("HCCL_HOST_SOCKET_PORT_RANGE", "60000-60050")
        env.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "61000-61050")
        if os.environ.get("ASCEND_USE_SHORT_CONNECTION") is not None:
            env["ASCEND_USE_SHORT_CONNECTION"] = os.environ["ASCEND_USE_SHORT_CONNECTION"]
    else:
        env.setdefault("NCCL_DEBUG", "WARN")

    rollout_lower = rollout_name.lower()
    if "vllm" in rollout_lower:
        env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
        env.setdefault("VLLM_USE_V1", "1")
        env.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

    return env


def _build_base_config(*, n_gpus_per_node: int, model_path: str) -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "+async_training.partial_rollout=True",
            ],
        )

    config.actor_rollout_ref.model.path = os.path.expanduser(model_path)
    config.actor_rollout_ref.rollout.name = os.environ["ROLLOUT_NAME"]
    config.actor_rollout_ref.rollout.max_num_seqs = 256
    config.actor_rollout_ref.rollout.response_length = 4096
    config.actor_rollout_ref.rollout.checkpoint_engine.backend = "nccl"
    config.actor_rollout_ref.rollout.nnodes = 1
    config.trainer.n_gpus_per_node = n_gpus_per_node
    config.trainer.nnodes = 1

    return config


async def _run_update_weights_with_global_steps_none_collect_token_ids(
    config: DictConfig,
    *,
    rollout_name: str,
) -> list[int]:
    ray.init(runtime_env={"env_vars": _ray_runtime_env_vars(rollout_name)})
    try:
        model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        checkpoint_engine_config: CheckpointEngineConfig = omega_conf_to_dataclass(
            config.actor_rollout_ref.rollout.checkpoint_engine
        )
        trainer_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node], max_colocate_count=3)
        trainer = create_trainer_worker_group(trainer_pool, model_config, checkpoint_engine_config)
        trainer.reset()

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

        await checkpoint_manager.update_weights(global_steps=None)
        prompt = [{"role": "user", "content": "How to make a sandwich?"}]
        prompt_ids = model_config.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
        output = await server_manager.generate(
            request_id="ci_compare_load_format",
            prompt_ids=prompt_ids,
            sampling_params={
                "temperature": 0.0,
                "top_p": 1.0,
                "logprobs": False,
            },
        )
        if output.stop_reason in ("aborted", "abort"):
            raise RuntimeError(f"generation aborted: stop_reason={output.stop_reason!r}")
        if output.extra_fields["global_steps"] is not None:
            raise RuntimeError(
                f"expected global_steps None, got {output.extra_fields['global_steps']!r}"
            )
        return list(output.token_ids)
    finally:
        ray.shutdown()


async def _async_main(*, n_gpus_per_node: int, model_path: str) -> int:
    rollout_name = os.environ["ROLLOUT_NAME"]
    base = _build_base_config(n_gpus_per_node=n_gpus_per_node, model_path=model_path)

    config_dummy_dtensor = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    config_dummy_dtensor.actor_rollout_ref.rollout.load_format = "dummy_dtensor"

    config_auto = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    config_auto.actor_rollout_ref.rollout.load_format = "auto"

    out_dummy = await _run_update_weights_with_global_steps_none_collect_token_ids(
        config_dummy_dtensor, rollout_name=rollout_name
    )
    out_auto = await _run_update_weights_with_global_steps_none_collect_token_ids(
        config_auto, rollout_name=rollout_name
    )

    if out_dummy != out_auto:
        print(
            "CI FAILED: token_ids differ between load_format=dummy_dtensor and load_format=auto "
            f"(len {len(out_dummy)} vs {len(out_auto)}).",
            file=sys.stderr,
        )
        return 1

    print("CI OK: token_ids match for dummy_dtensor vs auto after update_weights(global_steps=None).")
    return 0


def main() -> None:
    default_n = int(os.environ.get("VERL_N_GPUS_PER_NODE", "4"))
    default_model = os.environ.get("VERL_MODEL_PATH", "~/models/Qwen/Qwen3-VL-2B-Instruct")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-gpus-per-node",
        type=int,
        default=default_n,
        help="Trainer world size per node; rollout uses the same count in standalone mode "
        f"(needs >= 2x this many accelerators). Default: {default_n} (env VERL_N_GPUS_PER_NODE).",
    )
    parser.add_argument(
        "--model-path",
        default=default_model,
        help=f"HuggingFace model directory. Default from VERL_MODEL_PATH or {default_model!r}.",
    )
    args = parser.parse_args()

    if "ROLLOUT_NAME" not in os.environ:
        print("ERROR: ROLLOUT_NAME environment variable is not set.", file=sys.stderr)
        sys.exit(2)

    need_devices = 2 * args.n_gpus_per_node
    available = _accelerator_count()
    if available > 0 and available < need_devices:
        print(
            f"ERROR: this job needs at least {need_devices} accelerators "
            f"(trainer {args.n_gpus_per_node} + standalone rollout {args.n_gpus_per_node}), "
            f"but only {available} visible. Lower --n-gpus-per-node or fix visibility.",
            file=sys.stderr,
        )
        sys.exit(3)

    model_path = os.path.expanduser(args.model_path)
    code = asyncio.run(_async_main(n_gpus_per_node=args.n_gpus_per_node, model_path=model_path))
    sys.exit(code)


if __name__ == "__main__":
    main()
