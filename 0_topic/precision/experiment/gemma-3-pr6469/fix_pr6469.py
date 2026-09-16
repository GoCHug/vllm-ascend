"""Apply PR #6469 (commit 8e66299bd) fix to vllm-ascend attention_v1.py.

Usage (inside pod): python -u fix_pr6469.py
- backs up original to attention_v1.py.bak_pr6469
- asserts anchor uniqueness before replacing
- syntax-checks the patched file

Prerequisite: vllm-ascend v0.14.0rc1 (bug present, fix not backported).
After patch: rerun repro_pr6469.py and expect [RESULT] FIXED.
"""
import py_compile
import shutil

PATH = "/vllm-workspace/vllm-ascend/vllm_ascend/attention/attention_v1.py"
BACKUP = PATH + ".bak_pr6469"

src = open(PATH).read()
shutil.copyfile(PATH, BACKUP)
print("backup ->", BACKUP)

old_head = "        output, _ = torch_npu.npu_fused_infer_attention_score("
new_head = "        attn_output, _ = torch_npu.npu_fused_infer_attention_score("

old_tail = (
    "        output = output.view(batch_size, self.num_heads, self.head_size)\n"
    "        return output\n"
)
new_tail = (
    "        attn_output = attn_output.view(batch_size, self.num_heads, self.head_size)\n"
    "        output[:batch_size] = attn_output[:batch_size]\n"
    "        return output\n"
)

c_head = src.count(old_head)
c_tail = src.count(old_tail)
print("anchor count head =", c_head)
print("anchor count tail =", c_tail)
assert c_head == 1, f"head anchor not unique: {c_head}"
assert c_tail == 1, f"tail anchor not unique: {c_tail}"

src = src.replace(old_head, new_head)
src = src.replace(old_tail, new_tail)

open(PATH, "w").write(src)
print("PATCH_APPLIED")

py_compile.compile(PATH, doraise=True)
print("SYNTAX_OK")
