import torch

TRUE_DIR = '/a2_inference/itask/workdir/gch02599191/wc3ytlxq7ru781mo/code/true/step1/rank0/dump_tensor_data/'
FALSE_DIR = '/a2_inference/itask/workdir/gch02599191/wc3ytlxq7ru781mo/code/false/step1/rank0/dump_tensor_data/'

# 需要加载的 .pt 文件列表
files = [
    'Module.model.embed_tokens.AscendVocabParallelEmbedding.forward.0.input.0.pt',
]

for fname in files:
    true_path = TRUE_DIR + fname
    false_path = FALSE_DIR + fname

    print("=" * 80)
    print(f"文件: {fname}")
    print("=" * 80)

    t_true = torch.load(true_path, map_location=torch.device('cpu'))
    t_false = torch.load(false_path, map_location=torch.device('cpu'))

    print("--- TRUE ---")
    print(f"type: {type(t_true)}")
    if isinstance(t_true, torch.Tensor):
        print(f"shape: {t_true.shape}, dtype: {t_true.dtype}")
        print(t_true)
    else:
        print(t_true)

    print("--- FALSE ---")
    print(f"type: {type(t_false)}")
    if isinstance(t_false, torch.Tensor):
        print(f"shape: {t_false.shape}, dtype: {t_false.dtype}")
        print(t_false)
    else:
        print(t_false)

    print()
