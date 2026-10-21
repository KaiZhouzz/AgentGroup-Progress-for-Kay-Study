# 长租公寓运营 Agent 平台（Phase 1 Demo）

基于 LangChain + LangGraph 编排的长租公寓运营助手，统一对话入口覆盖三类场景：

| 场景 | 说明 | 是否可纯本地 |
|---|---|---|
| 📚 知识库问答 | 制度/流程类 RAG，检索 Chroma，回答带引用文件名 | ✅ |
| 📊 算法分析 | LLM 生成 pandas 代码 → 受限子进程沙箱执行 → 经营建议 | ✅（有云端 Key 时成功率更高） |
| ✍️ AIGC | 招租/社群文案生成（本期为 stub，单轮文案） | ✅ |

## 启动顺序

```bash
# 1. 安装依赖（项目内 .venv）
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. 生成样例数据（可选，可重复运行）
.venv/bin/python scripts/make_samples.py

# 3. 启动后端（FastAPI，默认 8000 端口）
.venv/bin/uvicorn app.main:app --reload

# 4. 启动前端（Streamlit，另开一个终端）
.venv/bin/streamlit run frontend/streamlit_app.py
```

打开浏览器访问 Streamlit 给出的地址（默认 http://localhost:8501）。

## 模型配置

复制 `.env.example` 为 `.env` 后按需修改：

| 变量 | 默认 | 说明 |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | 本地 Ollama 地址 |
| `CHAT_MODEL` | `qwen2.5:latest` | 本地对话模型 |
| `EMBED_MODEL` | `nomic-embed-text:latest` | 本地 embedding（RAG 专用） |
| `DASHSCOPE_API_KEY` | 空 | 阿里云百炼 Key；**留空则全部走本地** |
| `DASHSCOPE_MODEL` | `qwen-plus` | 云端模型（代码生成优先使用） |

### 降级行为（重要）

- **无 Docker**：代码沙箱自动使用受限子进程实现（临时目录 + 静态 import 黑名单 + resource 限制 + 硬超时 kill）。`app/sandbox/executor.py` 中预留了 `DockerExecutor` 接口骨架，未来有 Docker 环境后实现即可切换。
- **无 `DASHSCOPE_API_KEY`**：所有模型调用走本地 Ollama。对话/RAG/文案不受影响；**数据分析的代码生成会降级到本地 qwen2.5**，通过 prompt 中的 few-shot 示例提高成功率，但复杂分析的成功率低于云端模型，请尽量用明确的指标口径提问。
- **Ollama 未运行**：服务能启动，但一切模型调用会失败，`/health` 会显示 `ollama_online: false`。

## API

- `GET /health`：健康检查（Ollama 状态、是否配置了云端 Key、已上传文件列表）
- `POST /upload`：上传文件。`kind=auto|kb|data`；`kb` 解析入库 Chroma，`data` 登记为分析数据
- `POST /chat`：`{"question": "...", "session_id": "..."}` → `{"intent": "knowledge|analysis|aigc|clarify", "answer": "...", ...}`

## 目录说明

```
app/
├── main.py                # FastAPI 入口（/chat /upload /health）
├── core/
│   ├── config.py          # pydantic-settings 读 .env
│   └── model_router.py    # Ollama 本地 / DashScope 云端选择与回退
├── agents/
│   ├── router.py          # LangGraph：classify → knowledge/analysis/aigc（含 clarify 兜底）
│   ├── knowledge.py       # RAG：入库 + 检索 + 带引用回答
│   ├── analysis.py        # 代码生成 → 沙箱执行 → 经营建议
│   └── aigc.py            # 文案生成 stub
├── sandbox/
│   └── executor.py        # CodeExecutor 抽象 + SubprocessExecutor + DockerExecutor 占位
└── parsers/
    └── loader.py          # pdf/docx/xlsx/csv/txt → 文本块 / 数据概要
frontend/streamlit_app.py  # 统一对话窗口
data/samples/              # 样例数据（scripts/make_samples.py 生成）
data/chroma/               # Chroma 向量库存储（自动生成，已 gitignore）
```

## 路由逻辑简述

1. `classify` 节点先走规则层：命中"分析/欠费/出租率…"等关键词且有数据文件 → `analysis`；"文案/招租…" → `aigc`；"流程/制度/退租…" → `knowledge`。
2. 规则置信度 < 0.7 时调用 LLM 二次分类。
3. 综合置信度 < 0.45 时进入 `clarify`，向用户反问确认需求。

## 沙箱安全模型（Phase 1）

- 静态检查 import 白名单：仅允许 pandas/numpy/matplotlib/openpyxl 等数据分析库
- `resource` 软限制：CPU 时间、地址空间（内存）、输出文件大小、进程数
- 硬超时后 `killpg` 终止整组进程
- 数据文件以绝对路径白名单传入，脚本通过 `from data_files import FILES, OUT_DIR` 获取

> 注意：子进程沙箱是 macOS 无 Docker 下的权宜方案，隔离强度弱于容器，请勿在生产环境直接执行不可信代码。
