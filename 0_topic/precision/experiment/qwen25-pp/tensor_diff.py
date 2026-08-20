import torch

TRUE_DIR = '/true/step1/rank0/dump_tensor_data/'
FALSE_DIR = '/false/step1/rank0/dump_tensor_data/'

# 需要对比的 .pt 文件列表
files = [
    'Module.model.embed_tokens.AscendVocabParallelEmbedding.forward.0.input.0.pt',
    'Module.model.embed_tokens.AscendVocabParallelEmbedding.forward.0.output.0.pt',
    'Module.model.embed_tokens.AscendVocabParallelEmbedding.forward.0.parameters.weight.pt',
]


def load_tensor(path):
    """加载 .pt 文件，统一返回 dict (name -> Tensor) 以便对比。"""
    data = torch.load(path, map_location=torch.device('cpu'))
    if isinstance(data, torch.Tensor):
        return {'__main__': data}
    elif isinstance(data, dict):
        return {k: v for k, v in data.items() if isinstance(v, torch.Tensor)}
    elif isinstance(data, (list, tuple)):
        return {f'[{i}]': item for i, item in enumerate(data) if isinstance(item, torch.Tensor)}
    else:
        return {'__main__': data}


def compare_tensor(name, t_true, t_false):
    """对比两个 Tensor，打印差异统计。"""
    print(f"  [{name}]")
    if not isinstance(t_true, torch.Tensor) or not isinstance(t_false, torch.Tensor):
        print(f"    类型不匹配: true={type(t_true)}, false={type(t_false)}")
        return

    # 形状 / dtype
    same_shape = t_true.shape == t_false.shape
    same_dtype = t_true.dtype == t_false.dtype
    print(f"    shape: true={tuple(t_true.shape)}, false={tuple(t_false.shape)}, "
          f"{'一致' if same_shape else '不一致!'}")
    print(f"    dtype: true={t_true.dtype}, false={t_false.dtype}, "
          f"{'一致' if same_dtype else '不一致!'}")

    if not same_shape:
        print("    shape 不同，跳过数值对比")
        return

    # 统一 dtype 再做数值对比
    t_a = t_true.float()
    t_b = t_false.float()

    diff = (t_a - t_b).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    # 相对误差（避免除零）
    denom = t_a.abs().clamp(min=1e-8)
    rel_diff = (diff / denom)
    max_rel = rel_diff.max().item()
    mean_rel = rel_diff.mean().item()

    # 完全相同的比例
    exact_match = (diff == 0).float().mean().item()

    print(f"    max_abs_diff:  {max_diff:.6e}")
    print(f"    mean_abs_diff: {mean_diff:.6e}")
    print(f"    max_rel_diff:  {max_rel:.6e}")
    print(f"    mean_rel_diff: {mean_rel:.6e}")
    print(f"    exact_match_ratio: {exact_match:.4%}")

    # 差异最大的前 5 个位置
    flat_diff = diff.flatten()
    top_k = min(5, flat_diff.numel())
    top_vals, top_idx = flat_diff.topk(top_k)
    print(f"    top-{top_k} 差异位置 (abs_diff):")
    for i in range(top_k):
        idx = top_idx[i].item()
        v_true = t_a.flatten()[idx].item()
        v_false = t_b.flatten()[idx].item()
        print(f"      idx={idx}: true={v_true:.6e}, false={v_false:.6e}, "
              f"diff={top_vals[i].item():.6e}")


for fname in files:
    true_path = TRUE_DIR + fname
    false_path = FALSE_DIR + fname

    print("=" * 80)
    print(f"对比文件: {fname}")
    print("=" * 80)

    t_data = load_tensor(true_path)
    f_data = load_tensor(false_path)

    all_keys = sorted(set(list(t_data.keys()) + list(f_data.keys())))
    for key in all_keys:
        t_true = t_data.get(key, None)
        t_false = f_data.get(key, None)
        if t_true is None:
            print(f"  [{key}] 仅存在于 false 目录")
            continue
        if t_false is None:
            print(f"  [{key}] 仅存在于 true 目录")
            continue
        compare_tensor(key, t_true, t_false)
    print()
