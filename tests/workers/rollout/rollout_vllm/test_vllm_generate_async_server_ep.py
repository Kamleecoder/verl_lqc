import asyncio
import gc
import os
from uuid import uuid4

import pytest
import ray
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.utils.device import get_device_id, is_support_ipc
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import RolloutMode, TokenOutput
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
from verl.workers.rollout.vllm_rollout.utils import get_device_uuid

MODEL_PATH = "/data02/Moonlight-16B-A3B"

# Test with Moonlight-16B-A3B using Pure EP (Expert Parallelism) without TP
# This test uses expert_tensor_parallel_size for MoE models instead of tensor_model_parallel_size
# ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 pytest tests/workers/rollout/rollout_vllm/test_vllm_generate_async_server_ep.py -v -s


def _tokenize_prompt(text: str) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    messages = [{"role": "user", "content": text}]
    token_ids = normalize_token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    assert len(token_ids) > 0, "Prompt should produce at least one token."
    return token_ids


def _decode_tokens(token_ids: list[int]) -> str:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _build_configs(load_format: str, num_gpus: int = 4):
    rollout_cfg = OmegaConf.create(
        {
            "_target_": "verl.workers.config.RolloutConfig",
            "name": "vllm",
            "mode": "async",
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "gpu_memory_utilization": 0.85,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 64,
            "max_model_len": 4096,
            "dtype": "bfloat16",
            "load_format": load_format,
            "enforce_eager": False,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "enable_sleep_mode": False,
            "free_cache_engine": True,
            "disable_log_stats": True,
            "prompt_length": 512,
            "response_length": 256,
            "top_k": -1,
            "top_p": 1.0,
            "temperature": 0.0,
            "engine_kwargs": {
                "vllm": {
                    "additional_config": {
                        "ascend_compilation_config": {
                            "enable_npugraph_ex": True,
                        },
                        "expert_parallel_size": num_gpus,
                    },
                }
            },
        }
    )
    model_cfg = OmegaConf.create(
        {
            "_target_": "verl.workers.config.HFModelConfig",
            "path": MODEL_PATH,
            "trust_remote_code": True,
            "load_tokenizer": True,
        }
    )
    return rollout_cfg, model_cfg


def _start_server(load_format: str, force_dummy_after_init: bool = False, num_gpus: int = 4):
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

    rollout_cfg, model_cfg = _build_configs(load_format=load_format, num_gpus=num_gpus)

    cuda_visible_devices = ",".join(str(i) for i in range(num_gpus))
    ray_devices = ",".join(str(i) for i in range(num_gpus))

    server = ray.remote(vLLMHttpServer).options(
        runtime_env={
            "env_vars": {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": ray_devices,
                "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": ray_devices,
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
        gpus_per_node=num_gpus,
        nnodes=1,
        cuda_visible_devices=cuda_visible_devices,
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
            sampling_params={"max_tokens": 96, "temperature": 0.0, "top_p": 1.0, "top_k": -1},
            request_id=f"test_{tag}_{uuid4().hex[:8]}",
        ),
        timeout=300,
    )
    assert isinstance(output, TokenOutput)
    assert len(output.token_ids) > 0
    text = _decode_tokens(output.token_ids)
    assert text.strip() != ""
    return text


def _stop_server(server):
    if server is not None:
        ray.kill(server)


def _iter_reference_weights():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    try:
        for name, tensor in model.state_dict().items():
            yield name, tensor
    finally:
        del model
        gc.collect()


def _real_update_dummy_server_weights(server):
    device_uuid = get_device_uuid(get_device_id())
    zmq_handle = f"ipc:///tmp/rl-colocate-zmq-{device_uuid}.sock"
    update_ref = server.collective_rpc.remote(
        "update_weights_from_ipc",
        kwargs={"use_shm": not is_support_ipc()},
    )
    sender = BucketedWeightSender(
        zmq_handle=zmq_handle,
        bucket_size_mb=1024,
        use_shm=not is_support_ipc(),
    )
    asyncio.run(sender.async_send_weights(_iter_reference_weights()))
    ray.get(update_ref, timeout=1800)


def test_compare_dummy_update_and_auto_outputs_same_prompt_ep():
    """Test with Moonlight-16B-A3B using Pure EP (no TP, only Expert Parallelism)"""
    prompt = "写一段关于昇腾的介绍"
    num_gpus = 4  # Use 4 GPUs for EP

    dummy_server = None
    auto_server = None
    dummy_text = ""
    auto_text = ""
    try:
        dummy_server = _start_server(load_format="dummy", force_dummy_after_init=True, num_gpus=num_gpus)
        # Do real base-weight sync to make dummy comparable with auto.
        _real_update_dummy_server_weights(dummy_server)
        ray.get(dummy_server.set_global_steps.remote(1))
        dummy_text = _generate_text(dummy_server, prompt, "dummy_update")

        # Release resources from dummy server before starting auto server.
        _stop_server(dummy_server)
        dummy_server = None

        auto_server = _start_server(load_format="auto", force_dummy_after_init=False, num_gpus=num_gpus)
        auto_text = _generate_text(auto_server, prompt, "auto")

        print(
            "\n[Compare Prompt] "
            f"{prompt}\n[Generated][dummy+update] {dummy_text}\n[Generated][auto] {auto_text}\n"
        )
    finally:
        _stop_server(dummy_server)
        _stop_server(auto_server)
        if ray.is_initialized():
            ray.shutdown()
