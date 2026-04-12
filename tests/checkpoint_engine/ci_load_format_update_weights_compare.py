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

Run from the repository root (same expectations as ``test_special_server_adapter``):
``ROLLOUT_NAME`` must be set; GPUs and model path must match your environment.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import ray
from omegaconf import DictConfig, OmegaConf

from tests.checkpoint_engine.test_utils import create_trainer_worker_group
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AsyncLLMServerManager
from verl.single_controller.ray import RayResourcePool
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import CheckpointEngineConfig, HFModelConfig


def _build_base_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "+async_training.partial_rollout=True",
            ],
        )

    config.actor_rollout_ref.model.path = os.path.expanduser("~/models/Qwen/Qwen3-VL-2B-Instruct")
    config.actor_rollout_ref.rollout.name = os.environ["ROLLOUT_NAME"]
    config.actor_rollout_ref.rollout.max_num_seqs = 256
    config.actor_rollout_ref.rollout.response_length = 4096
    config.actor_rollout_ref.rollout.checkpoint_engine.backend = "nccl"
    config.actor_rollout_ref.rollout.nnodes = 1
    config.trainer.n_gpus_per_node = 4
    config.trainer.nnodes = 1

    return config


async def _run_update_weights_with_global_steps_none_collect_token_ids(config: DictConfig) -> list[int]:
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
                "VLLM_DISABLE_COMPILE_CACHE": "1",
            }
        }
    )
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


async def _async_main() -> int:
    base = _build_base_config()

    config_dummy_dtensor = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    config_dummy_dtensor.actor_rollout_ref.rollout.load_format = "dummy_dtensor"

    config_auto = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    config_auto.actor_rollout_ref.rollout.load_format = "auto"

    out_dummy = await _run_update_weights_with_global_steps_none_collect_token_ids(config_dummy_dtensor)
    out_auto = await _run_update_weights_with_global_steps_none_collect_token_ids(config_auto)

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if "ROLLOUT_NAME" not in os.environ:
        print("ERROR: ROLLOUT_NAME environment variable is not set.", file=sys.stderr)
        sys.exit(2)
    code = asyncio.run(_async_main())
    sys.exit(code)


if __name__ == "__main__":
    main()
