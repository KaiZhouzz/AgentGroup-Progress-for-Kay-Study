"""视频生成 agent：SiliconFlow Wan2.2 + 分镜尾帧续接，突破 5 秒限制。

流程：用户需求 → 本地 7B 拆成 N 个 5 秒镜头的分镜（免费）→ 镜头 1 走 T2V →
抽上一段最后一帧作为镜头 2+ 的首帧走 I2V → 本地 ffmpeg 串成一条成片。

任务异步执行：submit 立即返回 job_id，前端轮询 /video/status/{job_id}；
成片落在 data/videos/ 由 /video/file/{job_id} 提供播放。
"""
from __future__ import annotations

import base64
import logging
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)

SUBMIT_URL = "https://api.siliconflow.cn/v1/video/submit"
STATUS_URL = "https://api.siliconflow.cn/v1/video/status"
MODEL_T2V = "Wan-AI/Wan2.2-T2V-A14B"
MODEL_I2V = "Wan-AI/Wan2.2-I2V-A14B"

_video_dir = Path(settings.data_dir) / "videos"

# 内存态任务表：job_id → {status, step, total, phase, storyboard, file, error, created_at}
_jobs: dict[str, dict] = {}
_MAX_JOBS = 20

STORYBOARD_PROMPT = """把用户的视频需求拆成 {n} 个 5 秒镜头的分镜脚本（中文）。

用户是长租公寓运营人员。每个镜头是一段独立的视频提示词，包含"画面主体与场景 +
镜头运动 + 光线氛围"，镜头之间画面要连贯（像是在同一个空间里连续拍摄）。

重要：必须保留用户需求里的全部具体信息（房间特征、家具、物品、颜色、动作细节），
一个字都不要丢，按镜头分配下去；每镜头提示词 60~100 字，宁可长不可丢信息。

【用户需求】
{question}

严格按如下格式输出，每行一个镜头，不要输出任何其他内容：
镜头1：……
镜头2：……
"""

REWRITE_PROMPT = """把用户的视频需求改写成专业的视频生成提示词（中文）。

用户是长租公寓运营人员，需求通常口语化、带房源卖点。请改写成"画面主体 + 场景环境 +
镜头运动 + 光线氛围"结构的提示词，视频模型可直接据此生成。只输出提示词本身，
不要解释、不要引号、不超过 120 字。禁止编造需求中没有的具体信息（价格、地名、
奖项等）。

【用户需求】
{question}
"""


def _local_llm(prompt: str, temperature: float) -> str:
    from app.core import model_router

    llm = model_router.local_chat_model(temperature=temperature)
    return llm.invoke(prompt).content.strip()


def storyboard(question: str, shots: int) -> list[str]:
    """本地 7B 把需求拆成 shots 个镜头提示词；解析失败则整段需求当作单镜头。"""
    try:
        text = _local_llm(STORYBOARD_PROMPT.format(n=shots, question=question), temperature=0.3)
        prompts = [p.strip() for p in re.findall(r"镜头\s*\d+\s*[:：]\s*(.+)", text) if p.strip()]
        if prompts:
            return prompts[:shots]
    except Exception as exc:
        logger.warning("分镜生成失败，回退单镜头: %s", exc)
    return [question]


def rewrite(question: str) -> str:
    """（兼容旧调用）单镜头时的提示词润色。"""
    try:
        return _local_llm(REWRITE_PROMPT.format(question=question), temperature=0.5)
    except Exception as exc:
        logger.warning("视频提示词润色失败，原文直传: %s", exc)
        return question


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.siliconflow_api_key}",
        "Content-Type": "application/json",
    }


def _submit_sf(prompt: str, model: str, image_b64: str | None = None) -> str:
    """提交 SiliconFlow 任务，返回 requestId。"""
    payload: dict = {"model": model, "prompt": prompt, "image_size": "1280x720"}
    if image_b64:
        payload["image"] = image_b64
    resp = requests.post(SUBMIT_URL, json=payload, headers=_headers(), timeout=60)
    data = resp.json()
    if resp.status_code != 200:
        raise RuntimeError(f"提交失败({resp.status_code}): {data.get('message', data)}")
    task_id = (data.get("data") or {}).get("requestId") or data.get("requestId")
    if not task_id:
        raise RuntimeError(f"提交失败：响应中无 requestId: {data}")
    return task_id


def _wait_sf(task_id: str, timeout: int = 900) -> str:
    """轮询 SiliconFlow 任务直到完成，返回视频 url。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(10)
        r = requests.post(STATUS_URL, json={"requestId": task_id}, headers=_headers(), timeout=30)
        body = r.json()
        data = body.get("data") or body
        status = str(data.get("status", "")).lower()
        if status == "succeed":
            results = data.get("results") or {}
            videos = results.get("videos") if isinstance(results, dict) else results
            if videos and videos[0].get("url"):
                return videos[0]["url"]
            raise RuntimeError(f"任务完成但无视频地址: {data}")
        if status == "failed":
            raise RuntimeError(f"SiliconFlow 生成失败: {data.get('reason') or data.get('error') or data}")
    raise RuntimeError("等待 SiliconFlow 超时")


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _last_frame_jpg(mp4_path: Path) -> bytes:
    """抽取视频最后一帧（jpeg 字节），用作下一段 I2V 的首帧。"""
    out = mp4_path.with_suffix(".last.jpg")
    subprocess.run(
        [_ffmpeg(), "-y", "-sseof", "-0.1", "-i", str(mp4_path), "-update", "1", "-q:v", "2", str(out)],
        check=True, capture_output=True,
    )
    return out.read_bytes()


def _concat(parts: list[Path], out_path: Path) -> None:
    """同分辨率 mp4 顺序拼接（concat demuxer，不重编码）。"""
    lst = out_path.with_suffix(".txt")
    # concat 列表里的相对路径是相对于列表文件所在目录解析的，必须写绝对路径
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8")
    subprocess.run(
        [_ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-c", "copy", str(out_path)],
        check=True, capture_output=True,
    )
    lst.unlink(missing_ok=True)


def _run_job(job_id: str, question: str, shots: int, image_b64: str | None = None) -> None:
    job = _jobs[job_id]
    try:
        prompts = storyboard(question, shots)
        job["storyboard"] = prompts
        job["total"] = len(prompts)
        _video_dir.mkdir(parents=True, exist_ok=True)
        parts: list[Path] = []
        for i, prompt in enumerate(prompts):
            job["step"] = i + 1
            if i == 0:
                if image_b64:
                    job["phase"] = f"镜头 1/{len(prompts)}：以参考图为首帧生成中"
                    task = _submit_sf(prompt, model=MODEL_I2V, image_b64=image_b64)
                else:
                    job["phase"] = f"镜头 1/{len(prompts)}：文生视频生成中"
                    task = _submit_sf(prompt, model=MODEL_T2V)
            else:
                job["phase"] = f"镜头 {i + 1}/{len(prompts)}：抽取上一段尾帧"
                frame = _last_frame_jpg(parts[-1])
                job["phase"] = f"镜头 {i + 1}/{len(prompts)}：图生视频生成中"
                task = _submit_sf(
                    prompt, model=MODEL_I2V,
                    image_b64="data:image/jpeg;base64," + base64.b64encode(frame).decode(),
                )
            url = _wait_sf(task)
            shot_path = _video_dir / f"{job_id}_shot{i + 1}.mp4"
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            shot_path.write_bytes(r.content)
            parts.append(shot_path)
            logger.info("[video] job=%s shot=%d ok (%d bytes)", job_id, i + 1, len(r.content))
        final = _video_dir / f"{job_id}.mp4"
        if len(parts) > 1:
            job["phase"] = "拼接成片"
            _concat(parts, final)
        else:
            parts[0].rename(final)
        job["status"] = "succeed"
        job["phase"] = "完成"
        job["file"] = final.name
    except Exception as exc:
        logger.exception("[video] job=%s failed", job_id)
        job["status"] = "failed"
        job["error"] = str(exc)


def submit(question: str, shots: int = 3, image_b64: str | None = None) -> dict:
    """创建分镜续接生成任务，立即返回 {job_id}；失败返回 {error}。

    image_b64：参考图（data URI），提供时镜头 1 用 I2V 以该图为首帧。
    """
    if not settings.siliconflow_api_key.strip():
        return {"error": "未配置 SILICONFLOW_API_KEY"}
    if not (1 <= shots <= 4):
        return {"error": "镜头数需在 1~4 之间（每镜头约 0.29 美元，过长请分批生成）"}
    if len(_jobs) >= _MAX_JOBS:
        oldest = min(_jobs, key=lambda k: _jobs[k].get("created_at", 0))
        _drop_job(oldest)
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "status": "storyboard", "step": 0, "total": shots,
        "phase": "正在生成分镜", "storyboard": [], "file": None, "error": None,
        "question": question[:200], "created_at": time.time(),
    }
    threading.Thread(target=_run_job, args=(job_id, question, shots, image_b64), daemon=True).start()
    return {"job_id": job_id}


def job_status(job_id: str) -> dict | None:
    job = _jobs.get(job_id)
    if job is None:
        return None
    return {k: v for k, v in job.items() if k != "created_at"}


def list_jobs() -> list[dict]:
    """历史任务列表（新→旧），供"历史视频"入口展示。

    服务重启后内存任务清空，但成片仍在 data/videos/：扫描孤儿 mp4 一并列出，
    保证历史文件不丢。
    """
    out = []
    known_files = set()
    for job_id, job in _jobs.items():
        if job.get("file"):
            known_files.add(job["file"])
        out.append({
            "job_id": job_id,
            "status": job.get("status"),
            "question": job.get("question", ""),
            "file": job.get("file"),
            "created_at": job.get("created_at", 0),
        })
    for p in sorted(_video_dir.glob("*.mp4"), key=lambda x: -x.stat().st_mtime):
        if "_shot" in p.name or p.name in known_files:  # 分镜片段不是成片
            continue
        out.append({
                "job_id": p.stem, "status": "succeed",
                "question": "（服务重启前的历史文件）", "file": p.name,
                "created_at": p.stat().st_mtime,
            })
    return sorted(out, key=lambda j: -j["created_at"])


def job_file(job_id: str) -> Path | None:
    # 内存任务命中且成功 → 用记录里的文件名；否则按 job_id.mp4 兜底（重启前的历史文件）
    job = _jobs.get(job_id)
    name = job.get("file") if job and job.get("status") == "succeed" else None
    path = _video_dir / (name or f"{job_id}.mp4")
    return path if path.exists() else None


def _drop_job(job_id: str) -> None:
    _jobs.pop(job_id, None)
    for p in _video_dir.glob(f"{job_id}*"):
        p.unlink(missing_ok=True)
