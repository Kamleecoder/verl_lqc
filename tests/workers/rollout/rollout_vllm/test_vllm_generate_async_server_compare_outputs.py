import os
from pathlib import Path
from uuid import uuid4

import pytest
import ray
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


def _decode_tokens(token_ids: list[int]) -> str:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _build_configs(load_format: str):
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
            "load_format": load_format,
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
    return rollout_cfg, model_cfg


def _start_server(load_format: str, force_dummy_after_init: bool = False):
    runtime_env_vars = {
        "TOKENIZERS_PARALLELISM": "true",
        "VERL_LOGGING_LEVEL": "INFO",
        "VLLM_LOGGING_LEVEL": "INFO",
        "HCCL_CONNECT_TIMEOUT": os.environ.get("HCCL_CONNECT_TIMEOUT", "1500"),
        "HCCL_HOST_SOCKET_PORT_RANGE": os.environ.get("HCCL_HOST_SOCKET_PORT_RANGE", "60000-60050"),
        "HCCL_NPU_SOCKET_PORT_RANGE": os.environ.get("HCCL_NPU_SOCKET_PORT_RANGE", "61000-61050"),
    }
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": runtime_env_vars}, ignore_reinit_error=True)

    rollout_cfg, model_cfg = _build_configs(load_format=load_format)
    server = ray.remote(vLLMHttpServer).options(
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

    if force_dummy_after_init:
        # vLLMHttpServer normalizes dummy->auto in standalone mode; force it back for this comparison case.
        ray.get(server.__ray_call__.remote(lambda self: setattr(self.config, "load_format", "dummy")))

    ray.get(server.launch_server.remote())
    return server


def _generate_text(server, prompt: str, tag: str) -> str:
    prompt_ids = _tokenize_prompt(prompt)
    output = ray.get(
        server.generate.remote(
            prompt_ids=prompt_ids,
            sampling_params={"max_tokens": 96, "temperature": 0.7, "top_p": 0.9},
            request_id=f"test_{tag}_{uuid4().hex[:8]}",
        ),
        timeout=300,
    )
    assert isinstance(output, TokenOutput)
    assert len(output.token_ids) > 0
    text = _decode_tokens(output.token_ids)
    assert text.strip() != ""
    return text


def test_compare_dummy_update_and_auto_outputs_same_prompt():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model path does not exist: {MODEL_PATH}")

    prompt = "写一段关于昇腾的介绍"
    dummy_server = None
    auto_server = None
    try:
        dummy_server = _start_server(load_format="dummy", force_dummy_after_init=True)
        # Simulate "after update_weights" state marker for comparison output.
        ray.get(dummy_server.set_global_steps.remote(1))
        dummy_text = _generate_text(dummy_server, prompt, "dummy_update")

        auto_server = _start_server(load_format="auto", force_dummy_after_init=False)
        auto_text = _generate_text(auto_server, prompt, "auto")

        print(
            "\n[Compare Prompt] "
            f"{prompt}\n[Generated][dummy+update] {dummy_text}\n[Generated][auto] {auto_text}\n"
        )
    finally:
        if dummy_server is not None:
            ray.kill(dummy_server)
        if auto_server is not None:
            ray.kill(auto_server)
        if ray.is_initialized():
            ray.shutdown()
