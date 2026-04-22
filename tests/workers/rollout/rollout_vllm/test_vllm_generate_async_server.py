# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""
E2E test for vLLMHttpServer generate flow (text model).

Usage:
    pytest tests/workers/rollout/rollout_vllm/test_vllm_generate_async_server.py -v -s
    python tests/workers/rollout/rollout_vllm/test_vllm_generate_async_server.py
"""

import os
from pathlib import Path
from uuid import uuid4

import pytest
import ray
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import RolloutMode, TokenOutput
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

MODEL_PATH = Path(os.path.expanduser(os.environ.get("VERL_TEST_VLLM_MODEL_PATH", "~/models/Qwen/Qwen2.5-0.5B-Instruct")))


def _tokenize_prompt(text: str) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    messages = [{"role": "user", "content": text}]
    token_ids = normalize_token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    assert len(token_ids) > 0, "Prompt should produce at least one token."
    return token_ids


@pytest.fixture
def init_server():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model path does not exist: {MODEL_PATH}")

    runtime_env_vars = {
        "TOKENIZERS_PARALLELISM": "true",
        "VERL_LOGGING_LEVEL": "INFO",
        "VLLM_LOGGING_LEVEL": "INFO",
    }
    # NPU server-friendly defaults (harmless on CUDA environments if ignored).
    runtime_env_vars.update(
        {
            "HCCL_CONNECT_TIMEOUT": os.environ.get("HCCL_CONNECT_TIMEOUT", "1500"),
            "HCCL_HOST_SOCKET_PORT_RANGE": os.environ.get("HCCL_HOST_SOCKET_PORT_RANGE", "60000-60050"),
            "HCCL_NPU_SOCKET_PORT_RANGE": os.environ.get("HCCL_NPU_SOCKET_PORT_RANGE", "61000-61050"),
        }
    )

    ray.init(
        runtime_env={"env_vars": runtime_env_vars},
        ignore_reinit_error=True,
    )

    rollout_cfg = OmegaConf.create(
        {
            "_target_": "verl.workers.config.RolloutConfig",
            "name": "vllm",
            "mode": "async",
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "max_num_batched_tokens": 4096,
            "max_num_seqs": 128,
            "max_model_len": 2048,
            "dtype": "bfloat16",
            "load_format": "auto",
            "enforce_eager": True,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "enable_sleep_mode": False,
            "free_cache_engine": True,
            "disable_log_stats": True,
            "prompt_length": 1024,
            "response_length": 128,
            "top_k": -1,
            "top_p": 1.0,
            "temperature": 0.0,
        }
    )

    model_cfg = OmegaConf.create(
        {
            "_target_": "verl.workers.config.HFModelConfig",
            "path": str(MODEL_PATH),
            "trust_remote_code": True,
            "load_tokenizer": True,
        }
    )

    ServerCls = ray.remote(vLLMHttpServer)
    server = ServerCls.options(
        runtime_env={
            "env_vars": {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                "NCCL_CUMEM_ENABLE": "0",
            }
        },
        max_concurrency=16,
    ).remote(
        config=rollout_cfg,
        model_config=model_cfg,
        rollout_mode=RolloutMode.STANDALONE,
        workers=[],
        replica_rank=0,
        node_rank=0,
        gpus_per_node=1,
        nnodes=1,
        cuda_visible_devices="0",
    )

    ray.get(server.launch_server.remote())
    yield server
    ray.shutdown()


def test_generate(init_server):
    server = init_server
    prompt_ids = _tokenize_prompt("Write one short sentence about NPU inference.")

    output = ray.get(
        server.generate.remote(
            prompt_ids=prompt_ids,
            sampling_params={"max_tokens": 64, "temperature": 0.0, "top_p": 1.0},
            request_id=f"test_{uuid4().hex[:8]}",
        ),
        timeout=300,
    )
    assert isinstance(output, TokenOutput)
    assert isinstance(output.token_ids, list)
    assert len(output.token_ids) > 0
    assert output.stop_reason in ("completed", "aborted", None)


def test_generate_with_logprobs(init_server):
    server = init_server
    prompt_ids = _tokenize_prompt("Explain in one sentence what tensor parallelism is.")

    output = ray.get(
        server.generate.remote(
            prompt_ids=prompt_ids,
            sampling_params={"max_tokens": 64, "temperature": 0.0, "top_p": 1.0, "logprobs": True},
            request_id=f"test_lp_{uuid4().hex[:8]}",
        ),
        timeout=300,
    )
    assert isinstance(output, TokenOutput)
    assert len(output.token_ids) > 0
    assert output.log_probs is not None
    assert len(output.log_probs) > 0
    if isinstance(output.log_probs, torch.Tensor):
        assert output.log_probs.numel() > 0
    else:
        assert len(output.log_probs) == len(output.token_ids)


def test_generate_concurrent(init_server):
    server = init_server
    prompts = [
        "Describe a sunset over the ocean in one sentence.",
        "What is the purpose of attention mask?",
        "Give one benefit of mixed precision training.",
        "Why do we use async inference servers?",
    ]
    refs = [
        server.generate.remote(
            prompt_ids=_tokenize_prompt(text),
            sampling_params={"max_tokens": 64, "temperature": 0.0, "top_p": 1.0},
            request_id=f"concurrent_{i}_{uuid4().hex[:8]}",
        )
        for i, text in enumerate(prompts)
    ]
    outputs = ray.get(refs, timeout=600)
    assert len(outputs) == len(prompts)
    for out in outputs:
        assert isinstance(out, TokenOutput)
        assert len(out.token_ids) > 0
        assert out.stop_reason in ("completed", "aborted", None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
