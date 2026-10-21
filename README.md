# 业务运营 Agent 平台

基于 LangChain + LangGraph 编排的运营助手，统一对话入口覆盖多类场景：

| 场景 | 说明 | 是否可纯本地 |
|---|---|---|
| 📚 知识库问答 | 制度/流程类 RAG，检索 Chroma，回答带引用文件名；大小写不敏感匹配 + 按命中词数排名 | ✅ |
| 📊 数据分析 | LLM 生成 pandas 代码 → 受限子进程沙箱执行 → 经营建议；支持对话中直接上传 Excel 作为分析附件 | ✅（有云端 Key 时成功率更高） |
| ✍️ AIGC | 招租/社群文案生成 | ✅ |
| 🎬 视频生成 | 对话内识别视频需求 → 7B 模型拆分镜 → SiliconFlow T2V/I2V 逐镜头生成 → 尾帧续接 → ffmpeg 拼接成片；支持参考图（I2V） | ❌（需 SiliconFlow Key） |
| 📈 预测分析 | 预留意图口子（forecast），规划接入时间序列预测/欠费预警 | 规划中 |

## 服务架构

三个进程，全部可在一台机器上跑：

| 服务 | 端口 | 说明 |
|---|---|---|
| FastAPI 后端 | 8000 | `/chat` `/upload` 知识库/视频接口 |
| Streamlit 主对话 | 8501 | 统一对话窗口 + 历史视频 |
| Streamlit 知识库管理 | 8502 | 独立服务，登录（admin/admin）后可上传、查看切片、删除 |

## 启动顺序

```bash
# 1. 安装依赖（项目内 .venv）
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. 复制配置并修改
cp .env.example .env

# 3. 生成样例数据（可选，可重复运行）
.venv/bin/python scripts/make_samples.py

# 4. 启动后端（另开一个终端）
.venv/bin/uvicorn app.main:app --port 8000

# 5. 启动主对话前端
.venv/bin/streamlit run frontend/streamlit_app.py --server.port 8501

# 6. 启动知识库管理服务（可选）
.venv/bin/streamlit run frontend/kb_admin.py --server.port 8502
```

打开浏览器访问 Streamlit 给出的地址（默认 http://localhost:8501）。

## 模型配置

复制 `.env.example` 为 `.env` 后按需修改：

| 变量 | 默认 | 说明 |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | 本地 Ollama 地址 |
| `CHAT_MODEL` | `qwen2.5:latest` | 本地对话模型 |
| `CODE_MODEL` | `qwen2.5-coder:7b` | 代码生成专用模型（未拉取时回退 CHAT_MODEL） |
| `PLANNER_MODEL` | `qwen2.5:14b` | 语义理解/澄清裁判（未拉取时回退 CHAT_MODEL） |
| `EMBED_MODEL` | `nomic-embed-text:latest` | 本地 embedding（RAG 专用） |
| `DASHSCOPE_API_KEY` | 空 | 阿里云百炼 Key；**留空则全部走本地** |
| `DASHSCOPE_MODEL` | `qwen-plus` | 云端模型（代码生成优先使用） |
| `SILICONFLOW_API_KEY` | 空 | SiliconFlow Key；**视频生成必需** |
| `SILICONFLOW_VIDEO_MODEL` | `Wan-AI/Wan2.2-T2V-A14B` | 视频生成模型 |
| `CHROMA_HTTP_URL` | 空 | 远程 Chroma 服务地址；留空则用本地 `CHROMA_DIR` |

### 降级行为（重要）

- **无 Docker**：代码沙箱自动使用受限子进程实现（临时目录 + 静态 import 黑名单 + resource 限制 + 硬超时 kill）。`app/sandbox/executor.py` 中预留了 `DockerExecutor` 接口骨架，未来有 Docker 环境后实现即可切换。
- **无 `DASHSCOPE_API_KEY`**：所有模型调用走本地 Ollama。对话/RAG/文案不受影响；**数据分析的代码生成会降级到本地 qwen2.5**，通过 prompt few-shot 与确定性 import 注入提高成功率，复杂分析建议明确指标口径提问。
- **无 `SILICONFLOW_API_KEY`**：视频生成不可用，其余功能不受影响。
- **Ollama 未运行**：服务能启动，但一切模型调用会失败，`/health` 会显示 `ollama_online: false`。

## API

- `GET /health`：健康检查（Ollama 状态、云端 Key 配置、已上传文件列表）
- `POST /upload`：上传文件。`kind=auto|kb|data`；`kb` 解析入库 Chroma，`data` 登记为分析数据
- `POST /chat`：`{"question": "...", "session_id": "..."}` → `{"intent": "knowledge|analysis|aigc|video|clarify|forecast", "answer": "...", ...}`
- `POST /analysis/answer` / `POST /clarify/answer`：澄清轮追问与作答
- `POST /video/submit`：提交视频任务（需求文案 / 镜头数 / 可选参考图 base64）
- `GET /video/status/{job_id}` / `GET /video/file/{job_id}`：任务状态与成片下载
- `GET /video/list`：历史成片列表（含磁盘孤儿文件扫描）
- `GET /kb/sources` / `GET /kb/chunks` / `DELETE /kb/source`：知识库文件列表、切片详情、按文件删除

## 目录说明

```
app/
├── main.py                # FastAPI 入口
├── core/
│   ├── config.py          # pydantic-settings 读 .env
│   └── model_router.py    # Ollama 本地 / DashScope 云端选择与回退
├── agents/
│   ├── router.py          # LangGraph：classify → 各意图节点（含 clarify 兜底、forecast 预留）
│   ├── knowledge.py       # RAG：入库 + 检索 + 带引用回答
│   ├── analysis.py        # 代码生成 → 沙箱执行 → 经营建议
│   ├── pipeline.py        # 数据分析流水线（含确定性 import 注入兜底）
│   ├── aigc.py            # 文案生成
│   └── video.py           # 视频生成：分镜 → SiliconFlow T2V/I2V → 尾帧续接 → 拼接
├── sandbox/
│   └── executor.py        # CodeExecutor 抽象 + SubprocessExecutor + DockerExecutor 占位
└── parsers/
    └── loader.py          # pdf/docx/xlsx/csv/txt → 文本块 / 数据概要
frontend/streamlit_app.py  # 主对话（统一入口 + 对话内视频卡片 + 历史视频）
frontend/kb_admin.py       # 知识库管理（独立服务：登录/上传/切片/删除）
data/chroma/               # Chroma 向量库存储（自动生成，已 gitignore）
data/videos/               # 视频成片输出（已 gitignore）
```

## 路由逻辑简述

1. `classify` 节点先走规则层：命中"分析/欠费/出租率…"等关键词且有数据文件 → `analysis`；"视频/宣传片…" → `video`；"文案/招租…" → `aigc`；"流程/制度/退租…" → `knowledge`。
2. "推广/宣传"等模糊词会先反问"出文案还是做视频"，由用户选择。
3. 规则置信度 < 0.7 时调用 LLM 二次分类；综合置信度 < 0.45 时进入 `clarify` 反问确认。

## 沙箱安全模型

- 静态检查 import 白名单：仅允许 pandas/numpy/matplotlib/openpyxl 等数据分析库
- `resource` 软限制：CPU 时间、地址空间（内存）、输出文件大小、进程数
- 硬超时后 `killpg` 终止整组进程
- 数据文件以绝对路径白名单传入，脚本通过 `from data_files import FILES, OUT_DIR` 获取

> 注意：子进程沙箱是 macOS 无 Docker 下的权宜方案，隔离强度弱于容器，请勿在生产环境直接执行不可信代码。
