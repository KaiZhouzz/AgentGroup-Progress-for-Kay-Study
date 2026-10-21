"""FastAPI 入口：/chat 对话、/upload 文件上传（入库 RAG 并登记为可用数据）。"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agents import router as agent_router
from app.core.config import settings
from app.parsers import SUPPORTED_EXTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="长租公寓运营 Agent 平台", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _init_dirs():
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    # 从向量库同步已入库文件清单：内存态重启即清，但向量库是持久的，
    # 同步后重启的服务立刻知道知识库里有什么（否则 kb_sources 为空会影响路由回退）
    try:
        from app.agents.knowledge import _get_store

        res = _get_store()._collection.get(include=["metadatas"])
        _kb_sources.update(m["source"] for m in res["metadatas"] if m.get("source"))
        logger.info("知识库清单已同步: %d 个文件", len(_kb_sources))
    except Exception as exc:
        logger.warning("同步知识库清单失败（不影响启动）: %s", exc)


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"
    history: list[str] = []  # 最近几轮对话文本（含本轮之前），供分类器理解追问语境


class AnalysisAnswerRequest(BaseModel):
    clarify_id: str
    answers: dict[str, str]


class VideoSubmitRequest(BaseModel):
    question: str
    shots: int = 3  # 分镜数：每镜头 5 秒，>1 时尾帧续接 + 本地拼接
    image: str = ""  # 可选参考图（data:image/...;base64,...），作为镜头 1 首帧走 I2V


# 简单内存态：上传文件按会话隔离 {session_id: {filename: absolute_path}}。
# 旧会话的文件不会出现在新会话的分析上下文里，杜绝跨会话"资料搅合"。
# Phase 1 单进程够用。
_data_files: dict[str, dict[str, str]] = {}
_kb_sources: set[str] = set()

# 分析澄清会话：session_id → 可序列化的 AnalysisSession dict。内存态，1 小时未活跃清理。
_analysis_sessions: dict[str, dict] = {}
# 路由追问会话：session_id → {question}。用户选定回答方式后强制走对应链路。
_clarify_route_sessions: dict[str, dict] = {}
_ANALYSIS_SESSION_TTL = 3600


def _session_get(store: dict, session_id: str) -> dict:
    import time

    now = time.time()
    for sid in [s for s, v in store.items() if now - v.get("updated_at", 0) > _ANALYSIS_SESSION_TTL]:
        del store[sid]
    sess = store.get(session_id)
    if sess is None:
        raise HTTPException(404, "确认会话不存在或已处理，请重新提问")
    return sess


def data_file_list(session_id: str = "default") -> list[str]:
    return list(_data_files.get(session_id, {}).values())


@app.get("/health")
def health() -> dict:
    from app.core.model_router import ollama_available
    return {
        "ok": True,
        "ollama_online": ollama_available(),
        "cloud_key_configured": settings.has_cloud_key,
        "data_files": sorted({n for files in _data_files.values() for n in files}),
        "kb_sources": sorted(_kb_sources),
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...), kind: str = Form("auto"), session_id: str = Form("default"),
                 dept: str = Form("")):
    """上传文件：kind=kb 仅入库 RAG；kind=data 仅登记为分析数据；auto=两者都做。"""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTS:
        raise HTTPException(400, f"不支持的文件类型 {suffix}，支持: {sorted(SUPPORTED_EXTS)}")

    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.uploads_dir / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    result: dict = {"filename": file.filename, "kind": kind, "ingested_chunks": 0, "registered_data": False}

    # 表格文件默认只登记为分析数据，不进知识库（逐行文本化后切块 embedding 很慢，
    # 且表格走 RAG 检索效果差）；确有需要可在上传时显式选 kind=kb。
    from app.parsers import EXCEL_EXTS
    effective_kind = kind
    if kind == "auto" and suffix in EXCEL_EXTS:
        effective_kind = "data"

    if effective_kind in ("auto", "data"):
        _data_files.setdefault(session_id, {})[file.filename] = str(dest.resolve())
        result["registered_data"] = True

    if effective_kind in ("auto", "kb"):
        try:
            from app.agents.knowledge import ingest_file
            result["ingested_chunks"] = ingest_file(dest, dept=dept)
            if result["ingested_chunks"] == 0:
                result["kb_warning"] = "未从文件中提取到任何文本（可能是扫描件或空文件），知识库未新增内容"
                logger.warning("ingest 0 chunks: %s", file.filename)
            else:
                _kb_sources.add(file.filename)
        except Exception as exc:
            logger.exception("ingest failed")
            raise HTTPException(500, f"知识库入库失败: {exc}")

    logger.info("uploaded %s kind=%s (effective=%s)", file.filename, kind, effective_kind)
    return result


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    if not req.question.strip():
        raise HTTPException(400, "question 不能为空")
    logger.info("[chat] session=%s q=%r", req.session_id, req.question[:80])
    result = agent_router.handle_question(
        req.question,
        data_files=data_file_list(req.session_id),
        kb_sources=sorted(_kb_sources),
        history=req.history,
    )
    result["session_id"] = req.session_id
    # 确认类会话：会话状态留在后端，clarify_id 供对应 answer 端点找回
    sess = result.pop("session", None)
    if sess:
        sess["frontend_sid"] = req.session_id  # 找回该会话当时的数据文件范围
        if result.get("intent") == "clarify_route":
            _clarify_route_sessions[sess["session_id"]] = sess
        else:
            _analysis_sessions[sess["session_id"]] = sess
        result["clarify_id"] = sess["session_id"]
    return result


@app.post("/analysis/answer")
def analysis_answer(req: AnalysisAnswerRequest) -> dict:
    """回答分析澄清问题：找回会话 → 合并答案 → 继续执行流水线。"""
    from app.agents.pipeline import AnalysisPipeline, AnalysisSession

    sess = _session_get(_analysis_sessions, req.clarify_id)
    pipeline = AnalysisPipeline(sess["data_files"])
    result = pipeline.resume(AnalysisSession.from_dict(sess), req.answers)
    new_state = pipeline_result_session(result)
    if new_state:
        _analysis_sessions[new_state["session_id"]] = new_state
        result["clarify_id"] = new_state["session_id"]
    else:
        _analysis_sessions.pop(req.clarify_id, None)  # 已出最终结果，清理会话
    result.setdefault("confidence", 0.9)  # 追问是已分类请求的延续，置信度沿用
    return result


@app.post("/clarify/answer")
def clarify_route_answer(req: AnalysisAnswerRequest) -> dict:
    """回答路由追问：用户选定回答方式后，强制走对应链路重新执行。"""
    sess = _session_get(_clarify_route_sessions, req.clarify_id)
    choice = (req.answers.get("route") or "").strip()
    fsid = sess.pop("frontend_sid", "default")  # 该追问发起时前端会话的数据文件范围
    _clarify_route_sessions.pop(req.clarify_id, None)

    if "数据" in choice or "统计" in choice:
        force = "analysis"
    elif "知识库" in choice or "文档" in choice:
        force = "knowledge"
    elif "视频" in choice:
        force = "video"
    elif "文案" in choice:
        force = "aigc"
    else:
        # 自填内容：当作新的完整问题重新路由
        question = choice or sess["question"]
        return agent_router.handle_question(
            question,
            data_files=data_file_list(fsid),
            kb_sources=sorted(_kb_sources),
        )

    logger.info("[clarify_route] 用户选择 %s → %s", force, sess["question"][:50])
    return agent_router.handle_question(
        sess["question"],
        data_files=data_file_list(fsid),
        kb_sources=sorted(_kb_sources),
        force_intent=force,
    )


def pipeline_result_session(result: dict) -> dict | None:
    """clarify_analysis 结果里带着新的会话状态；最终结果没有。"""
    return result.pop("session", None)


@app.post("/video/submit")
def video_submit(req: VideoSubmitRequest) -> dict:
    """提交视频生成任务（分镜 + 尾帧续接）：后台异步执行，返回 job_id 供轮询。

    shots=1 为单条 5 秒视频；shots>1 时第 1 镜头文生视频、后续镜头用上一段
    尾帧做首帧图生视频，最后本地 ffmpeg 拼接成一条成片。
    """
    from app.agents import video

    if not req.question.strip():
        raise HTTPException(400, "question 不能为空")
    result = video.submit(req.question, shots=req.shots, image_b64=req.image or None)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.get("/kb/sources")
def kb_sources() -> dict:
    """知识库清单：按文件聚合（切片数、部门）。"""
    from app.agents import knowledge

    return {"sources": knowledge.list_sources()}


@app.get("/kb/chunks")
def kb_chunks(source: str) -> dict:
    """某文件的全部切片（id + 预览 + 部门）。"""
    from app.agents import knowledge

    return {"source": source, "chunks": knowledge.list_chunks(source)}


@app.delete("/kb/source")
def kb_delete(source: str) -> dict:
    """按文件删除知识库内容（该文件的全部切片）。"""
    from app.agents import knowledge

    removed = knowledge.delete_source(source)
    _kb_sources.discard(source)
    return {"source": source, "removed_chunks": removed}


@app.get("/video/list")
def video_list() -> dict:
    """历史视频任务列表（内存态：服务重启后清空，但文件仍在 data/videos/）。"""
    from app.agents import video

    return {"jobs": video.list_jobs()}


@app.get("/video/status/{job_id}")
def video_status(job_id: str) -> dict:
    """查询视频任务进度：storyboard/generating/succeed/failed + 当前镜头进度。"""
    from app.agents import video

    status = video.job_status(job_id)
    if status is None:
        raise HTTPException(404, "任务不存在或已清理")
    return status


@app.get("/video/file/{job_id}")
def video_file(job_id: str):
    """下载/播放成片（仅 succeed 后可用）。"""
    from fastapi.responses import FileResponse

    from app.agents import video

    path = video.job_file(job_id)
    if path is None:
        raise HTTPException(404, "视频不存在或尚未生成完成")
    return FileResponse(path, media_type="video/mp4", filename=f"video_{job_id}.mp4")
