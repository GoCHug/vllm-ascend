# skill 源码库

这个文件夹是**自己的 skill 源文件收藏库**——集中编写、整理、沉淀 skill 的地方，可随知识库一起备份 / 同步 / 分享。

## 机制说明（重要）

TRAE **只自动识别工作区根目录下的 `.trae/skills/`**：

```
C:\Users\89517\Desktop\vllm同步\.trae\skills\<skill名>\SKILL.md   ← 生效位置（已安装）
C:\Users\89517\Desktop\vllm同步\vllm-ascend\0_topic\precision\skill\<skill名>\SKILL.md   ← 本库（源文件，不生效）
```

放在本库的 skill **不会自动生效**——需要「安装」，即**复制到 `.trae/skills/` 同名目录下**。

## 安装方式

### 方式一：一键安装全部（推荐）

在工作区根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File "vllm-ascend\0_topic\precision\skill\install.ps1"
```

脚本会把本库下**所有** skill 同步到 `.trae/skills/`（覆盖更新），并列出安装结果。

### 方式二：手动安装单个

```powershell
Copy-Item "vllm-ascend\0_topic\precision\skill\<skill名>" "C:\Users\89517\Desktop\vllm同步\.trae\skills\" -Recurse -Force
```

## 日常使用流程

1. **新建/修改 skill**：在本库对应文件夹里编辑 `SKILL.md`；
2. **安装**：运行 `install.ps1`（或手动复制）；
3. **生效验证**：新开会话，skill 会出现在可用列表中；说「收集精度问题」「整理案例」等触发词即可验证。

## 目录约定

```
skill/
├── README.md                        ← 本说明
├── install.ps1                      ← 一键安装脚本
└── <skill名>/                       ← 每个 skill 一个文件夹
    └── SKILL.md                     ← skill 本体（frontmatter + 正文）
```

新增 skill 时：在本库建 `<skill名>/SKILL.md`（frontmatter 必须含 `name` 和 `description` 两个字段），然后跑一次 `install.ps1`。

## 当前 skill 清单

| skill | 功能 | 状态 |
|-------|------|------|
| `vllm-precision-tracker` | 收集更新 vllm/vllm-ascend 精度问题、整理可复现案例（三条工作流 + 五段制模板 + 硬性规则） | 已创建并安装 |
