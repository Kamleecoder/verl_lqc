import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from verl.utils.device import get_device_name


def _compare_state_dicts(state_dict1: dict, state_dict2: dict, atol: float = 1e-5, rtol: float = 1e-8) -> bool:
    """比较两个 state_dict 是否完全一致"""
    if state_dict1.keys() != state_dict2.keys():
        print("[ERROR] State dict keys mismatch!")
        return False

    all_close = True
    mismatch_count = 0
    for key in state_dict1.keys():
        tensor1 = state_dict1[key]
        tensor2 = state_dict2[key]
        
        if tensor1.shape != tensor2.shape:
            print(f"[ERROR] Shape mismatch for {key}")
            all_close = False
            continue
        
        if not torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol):
            max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
            if mismatch_count < 5:  # 只打印前5个不匹配的
                print(f"[ERROR] Value mismatch for {key}, max diff: {max_diff:.10f}")
            mismatch_count += 1
            all_close = False

    if mismatch_count > 0:
        print(f"[INFO]   总共有 {mismatch_count} 个张量不匹配")

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


def _do_forward_update(model, tokenizer, device: str, model_name: str):
    """
    对指定模型执行前向更新
    """
    print(f"\n[INFO] ========== 对 {model_name} 执行前向更新 ==========")
    
    model.train()
    
    # 准备输入
    prompt = "你好，请介绍一下你自己。"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 前向传播
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(**inputs, labels=inputs["input_ids"])
    
    loss = outputs.loss
    print(f"[INFO]   前向传播完成，Loss: {loss.item():.6f}")
    
    # 反向传播 + 参数更新
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"[INFO]   参数更新完成！")
    print(f"[INFO] ========== {model_name} 前向更新结束 ==========")
    
    return model.state_dict()


def test_two_different_starting_points_separate_update():
    """
    按你要求的测试：
    1. Model A: load_format=auto（真实权重，起点1）
    2. Model B: load_format=dummy（随机初始化，起点2）
    3. 不使用 load_state_dict 同步初始权重
    4. 对两个模型分别做前向更新
    5. 对比更新后的权重
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
    print(f"\n" + "="*80)
    print(f"[INFO] 测试设置：")
    print(f"[INFO]   Model A: load_format=auto（真实权重，起点1）")
    print(f"[INFO]   Model B: load_format=dummy（随机初始化，起点2）")
    print(f"[INFO]   不使用 load_state_dict 同步初始权重")
    print(f"[INFO]   对两个模型分别做前向更新")
    print(f"[INFO]   对比更新后的权重")
    print(f"="*80)

    # ================= 1. 加载 Tokenizer =================
    print("\n[1/6] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ================= 2. 加载 Model A (load_format=auto) =================
    print("\n[2/6] 加载 Model A (load_format=auto)...")
    model_a, state_dict_a_initial = _load_model(model_path, "auto", device)
    print(f"[INFO]   Model A 参数量: {sum(p.numel() for p in model_a.parameters())/1e9:.2f}B")

    # ================= 3. 加载 Model B (load_format=dummy) =================
    print("\n[3/6] 加载 Model B (load_format=dummy)...")
    model_b, state_dict_b_initial = _load_model(model_path, "dummy", device)
    print(f"[INFO]   Model B 参数量: {sum(p.numel() for p in model_b.parameters())/1e9:.2f}B")

    # ================= 4. 验证初始权重不同 =================
    print("\n[4/6] 验证初始状态...")
    initial_match = _compare_state_dicts(state_dict_a_initial, state_dict_b_initial)
    if initial_match:
        print("[WARNING] ⚠️ Model A 和 B 初始权重居然一致！")
    else:
        print("[INFO] ✅ Model A (auto) 和 B (dummy) 初始权重不同，符合预期")

    # ================= 5. 对两个模型分别做前向更新 =================
    print("\n[5/6] 对两个模型分别做前向更新...")
    
    state_dict_a_updated = _do_forward_update(model_a, tokenizer, device, "Model A (auto)")
    state_dict_b_updated = _do_forward_update(model_b, tokenizer, device, "Model B (dummy)")

    # ================= 6. 对比更新后的权重 =================
    print("\n[6/6] 最终对比：两个模型更新后的权重...")
    print("\n" + "="*80)
    print("[INFO] 预期说明：")
    print("[INFO]   因为 Model A 和 B 的初始权重完全不同（一个真实权重，一个随机初始化），")
    print("[INFO]   即使做完全相同的前向更新，最终的权重也肯定不同。")
    print("[INFO]   这是正常的、预期的结果。")
    print("="*80)
    
    final_match = _compare_state_dicts(state_dict_a_updated, state_dict_b_updated)
    
    print("\n" + "="*80)
    if final_match:
        print("[INFO] ⚠️  意外：更新后的权重居然完全一致！")
        print("[INFO] ⚠️  这不符合预期（因为初始权重不同）")
    else:
        print("[INFO] ✅ 结果符合预期：")
        print("[INFO] ✅   Model A (auto) 和 Model B (dummy)")
        print("[INFO] ✅   因为初始起点不同，更新后的权重也不同")
    print("="*80 + "\n")

    # 清理
    del model_a, model_b
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "npu":
        torch.npu.empty_cache()


if __name__ == "__main__":
    test_two_different_starting_points_separate_update()