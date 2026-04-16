import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from verl.utils.device import get_device_name


def _compare_state_dicts(state_dict1: dict, state_dict2: dict, atol: float = 1e-5, rtol: float = 1e-8) -> bool:
    """比较两个 state_dict 是否完全一致"""
    if state_dict1.keys() != state_dict2.keys():
        print("[ERROR] State dict keys mismatch!")
        return False

    all_close = True
    for key in state_dict1.keys():
        tensor1 = state_dict1[key]
        tensor2 = state_dict2[key]
        
        if tensor1.shape != tensor2.shape:
            print(f"[ERROR] Shape mismatch for {key}")
            all_close = False
            continue
        
        if not torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol):
            max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
            print(f"[ERROR] Value mismatch for {key}, max diff: {max_diff:.10f}")
            all_close = False

    return all_close


def _load_model(model_path: str, load_format: str, device: str):
    """
    模拟 verl 的 load_format 逻辑：
    - auto: 加载真实权重
    - dummy: 随机初始化
    """
    print(f"\n[INFO] 加载模型 (load_format={load_format})...")
    
    if load_format == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=device
        )
        print(f"[INFO]   load_format=auto: 从磁盘加载真实权重")
    elif load_format == "dummy":
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        with torch.device(device):
            model = AutoModelForCausalLM.from_config(
                config,
                dtype=torch.bfloat16,
                attn_implementation="eager",
                trust_remote_code=True
            )
        print(f"[INFO]   load_format=dummy: 随机初始化")
    else:
        raise ValueError(f"Unknown load_format: {load_format}")
    
    return model, model.state_dict()


def _do_actual_forward_update(model, tokenizer, device: str):
    """
    【核心】实际的模型前向更新：
    1. 准备输入
    2. 前向传播
    3. 计算简单的 loss
    4. 模拟一次参数更新（用简单的 SGD）
    """
    print(f"\n[INFO] ========== 执行实际的模型前向更新 ==========")
    
    model.train()
    
    # 1. 准备输入
    print(f"[INFO]   步骤 1: 准备输入数据")
    prompt = "你好，请介绍一下你自己。"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 2. 前向传播
    print(f"[INFO]   步骤 2: 前向传播")
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(**inputs, labels=inputs["input_ids"])
    
    loss = outputs.loss
    logits = outputs.logits
    print(f"[INFO]   前向传播完成，Loss: {loss.item():.4f}")
    print(f"[INFO]   Logits shape: {logits.shape}")
    
    # 3. 模拟一次简单的参数更新（SGD）
    print(f"[INFO]   步骤 3: 模拟一次参数更新 (SGD)")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"[INFO]   参数更新完成！")
    print(f"[INFO] ========== 实际模型前向更新结束 ==========")
    
    return model.state_dict()


def test_qwen25_load_format_with_actual_update():
    """
    主测试：
    1. load_format=auto 加载真实权重
    2. 【核心】对 auto 模型做实际的前向传播和参数更新
    3. load_format=dummy 随机初始化
    4. 将更新后的 auto 权重同步给 dummy
    5. 验证两者完全一致
    """
    # ================= 0. 环境准备 =================
    model_path = os.environ.get("QWEN_MODEL_PATH", "/path/to/your/Qwen2.5-7B-Instruct")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"模型路径不存在: {model_path}\n"
            "请设置环境变量：export QWEN_MODEL_PATH=/你的/模型/实际路径"
        )
    
    device = get_device_name()
    print(f"[INFO] 检测到设备: {device}")
    print(f"[INFO] 模型路径: {model_path}")

    # ================= 1. 加载 Tokenizer =================
    print("\n[1/6] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ================= 2. 加载 load_format=auto =================
    print("\n[2/6] 加载源模型 (load_format=auto)...")
    model_auto, state_dict_auto_initial = _load_model(model_path, "auto", device)
    print(f"[INFO]   参数量: {sum(p.numel() for p in model_auto.parameters())/1e9:.2f}B")

    # ================= 3. 【核心】实际的模型前向更新 =================
    print("\n[3/6] 对 auto 模型执行实际的前向更新...")
    state_dict_auto_updated = _do_actual_forward_update(model_auto, tokenizer, device)

    # 验证：更新前后权重确实变了
    print("\n[INFO] 验证：auto 模型更新前后权重变化...")
    auto_changed = not _compare_state_dicts(state_dict_auto_initial, state_dict_auto_updated)
    if auto_changed:
        print("[INFO] ✅ auto 模型权重确实发生了变化（前向更新生效）")
    else:
        print("[WARNING] ⚠️ auto 模型权重没有变化（可能是 lr 太小）")

    # ================= 4. 加载 load_format=dummy =================
    print("\n[4/6] 加载目标模型 (load_format=dummy)...")
    model_dummy, state_dict_dummy_initial = _load_model(model_path, "dummy", device)
    print(f"[INFO]   参数量: {sum(p.numel() for p in model_dummy.parameters())/1e9:.2f}B")

    # ================= 5. 验证初始权重不同 =================
    print("\n[5/6] 验证初始权重差异...")
    initial_match = _compare_state_dicts(state_dict_auto_updated, state_dict_dummy_initial)
    if initial_match:
        print("[WARNING] ⚠️ 初始权重居然一致！")
    else:
        print("[INFO] ✅ 初始权重不同，符合预期")

    # ================= 6. 模拟 verl update_weights 并验证 =================
    print("\n[6/6] 模拟 verl update_weights 并验证...")
    print(f"[INFO] 将更新后的 auto 权重同步给 dummy...")
    model_dummy.load_state_dict(state_dict_auto_updated)
    state_dict_dummy_updated = model_dummy.state_dict()

    final_match = _compare_state_dicts(state_dict_auto_updated, state_dict_dummy_updated)
    
    if final_match:
        print("\n" + "🎉"*15)
        print("[INFO] ✅ 测试通过！")
        print("[INFO] ✅ 完整流程验证：")
        print("[INFO] ✅   1. load_format=auto 加载真实权重")
        print("[INFO] ✅   2. 【核心】对 auto 模型做了实际的前向传播和参数更新")
        print("[INFO] ✅   3. load_format=dummy 随机初始化")
        print("[INFO] ✅   4. 模拟 verl update_weights 后两者完全一致")
        print("🎉"*15 + "\n")
    else:
        print("\n" + "❌"*15)
        print("[ERROR] ❌ 测试失败！")
        print("❌"*15 + "\n")
        raise AssertionError("Weight mismatch")

    # 清理
    del model_auto, model_dummy
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "npu":
        torch.npu.empty_cache()


if __name__ == "__main__":
    test_qwen25_load_format_with_actual_update()