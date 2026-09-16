import sys
import traceback

import torch
import torch_npu  # noqa: F401
torch.npu.set_device(0)

from types import SimpleNamespace
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

print("vllm_ascend file:", sys.modules["vllm_ascend"].__file__)

# --- config: gpt-oss-like SWA decode (sliding_window=128) ---
B          = 1
NUM_HEADS  = 16
NUM_KV     = 4
HEAD       = 64
SW         = 128    # sliding window (gpt-oss style)
BLOCK      = 128
NUM_BLOCKS = 8
SEQ_KV     = 500    # > SW so the window logic is exercised

try:
    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    impl.num_heads      = NUM_HEADS
    impl.num_kv_heads   = NUM_KV
    impl.head_size      = HEAD
    impl.sliding_window = SW
    impl.scale          = HEAD ** -0.5

    kv_shape = (NUM_BLOCKS, BLOCK, NUM_KV, HEAD)
    impl.key_cache   = torch.randn(kv_shape, dtype=torch.float16, device="npu")
    impl.value_cache = torch.randn(kv_shape, dtype=torch.float16, device="npu")

    n_blocks    = (SEQ_KV + BLOCK - 1) // BLOCK
    block_table = torch.arange(n_blocks, dtype=torch.int32, device="npu").unsqueeze(0)
    seq_lens    = torch.full((B,), SEQ_KV, dtype=torch.int64, device="npu")
    meta = SimpleNamespace(seq_lens=seq_lens, block_tables=block_table)

    # NOTE: real call chain (vllm attention/layer.py) passes query/output as
    # 3D [num_tokens, num_heads, head_size] per layer.py view convention.
    query  = torch.randn(B, NUM_HEADS, HEAD, dtype=torch.float16, device="npu")
    output = torch.full((B, NUM_HEADS, HEAD), float("nan"), dtype=torch.float16, device="npu")

    ret = impl._forward_fia_slidingwindow(query, meta, output)
    torch.npu.synchronize()

    buf_written = not torch.isnan(output).any().item()
    ret_valid   = not torch.isnan(ret).any().item()
    same_obj    = ret is output

    print("[JUDGE] ret is output (same object) :", same_obj)
    print("[JUDGE] output buffer written back  :", buf_written)
    print("[JUDGE] return tensor valid (no NaN):", ret_valid)

    if same_obj and buf_written:
        print("[RESULT] FIXED: out parameter written back in-place")
    elif (not same_obj) and (not buf_written):
        print("[RESULT] BUG CONFIRMED: out parameter NOT written back (buffer still NaN) - PR #6469 bug reproduced")
    else:
        print("[RESULT] UNEXPECTED state, inspect manually")
except Exception:
    traceback.print_exc()
print("REPRO_DONE")
