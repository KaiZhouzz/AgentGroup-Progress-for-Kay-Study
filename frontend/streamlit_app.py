"""长租公寓运营 Agent 平台 - 统一对话前端。

启动：streamlit run frontend/streamlit_app.py
视图：💬 对话（多轮对话 + 附件上传 + 对话内视频生成卡片）/ 📚 知识库管理（上传 + 部门预留）
"""
from __future__ import annotations

import base64
import io
import os
import time
import uuid
from pathlib import Path

import requests
import streamlit as st

# 外部网络访问时通过环境变量指向后端隧道地址；本地默认 localhost
API_BASE = os.environ.get("KT_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="长租公寓运营 Agent 平台", page_icon="🏠", layout="wide")
st.title("🏠 长租公寓运营 Agent 平台")

INTENT_LABELS = {
    "knowledge": "📚 知识库问答",
    "analysis": "📊 数据分析",
    "aigc": "✍️ 文案生成",
    "video": "🎬 视频生成",
    "forecast": "🔮 预测分析",
    "clarify": "❓ 需要澄清",
    "clarify_analysis": "🔍 需求确认",
    "clarify_route": "🔀 选择回答方式",
}
EXCEL_EXTS = {".xlsx", ".xls", ".csv"}
# 部门权限预留：上传时存入向量库元数据，检索过滤后续版本再加
DEPT_OPTIONS = ["不限", "运营部", "客服部", "工程部", "财务部", "人事部"]


def _new_session() -> str:
    sid = uuid.uuid4().hex[:8]
    st.session_state.messages[sid] = []
    return sid


# ---------- session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = {}
if "session_id" not in st.session_state:
    st.session_state.session_id = _new_session()
if "view" not in st.session_state:
    st.session_state.view = "💬 对话"
if "uploads" not in st.session_state:
    st.session_state.uploads = []
if "pending_clarify" not in st.session_state:
    st.session_state.pending_clarify = None
if "clarify_submitting" not in st.session_state:
    st.session_state.clarify_submitting = False

sid = st.session_state.session_id
msgs = st.session_state.messages[sid]


def render_result(result: dict) -> None:
    """渲染分析结果（图片/代码/输出），主提问和澄清提交共用。"""
    if result.get("images"):
        for img_b64 in result["images"]:
            st.image(io.BytesIO(base64.b64decode(img_b64)), use_container_width=True)
    if result.get("code"):
        with st.expander("查看生成的分析代码"):
            st.code(result["code"], language="python")
        if result.get("stdout"):
            with st.expander("查看执行输出"):
                st.code(result["stdout"], language="text")
        if result.get("stderr"):
            with st.expander("查看报错信息"):
                st.code(result["stderr"], language="text")


def render_video_card(m: dict, midx: int) -> None:
    """对话内视频生成卡片：未提交→表单；已提交→轮询进度；完成→内嵌播放。"""
    job_id = m.get("video_job")
    if not job_id:
        with st.container(border=True):
            st.markdown("**🎬 视频生成**")
            need = st.text_area(
                "画面需求",
                value=m.get("video_origin", ""),
                key=f"vq_{midx}",
                placeholder="例如：清晨阳光照进整洁的一居室客厅，绿植、原木家具，镜头缓缓推进到窗边",
                height=68,
                label_visibility="collapsed",
            )
            c1, c2 = st.columns([1, 2])
            shots = c1.number_input("镜头数（5 秒/镜头）", min_value=1, max_value=4, value=2, key=f"vs_{midx}")
            img = c2.file_uploader("参考图（可选，作为视频起始画面）", type=["png", "jpg", "jpeg", "webp"],
                                   key=f"vi_{midx}")
            if st.button("🎬 提交生成", key=f"vsubmit_{midx}", disabled=not need.strip()):
                image_b64 = ""
                if img is not None:
                    image_b64 = "data:image/jpeg;base64," + base64.b64encode(img.getvalue()).decode()
                with st.spinner("正在提交任务……"):
                    resp = requests.post(
                        f"{API_BASE}/video/submit",
                        json={"question": need, "shots": int(shots), "image": image_b64},
                        timeout=120,
                    )
                    resp.raise_for_status()
                m["video_job"] = resp.json()["job_id"]
                m["video_need"] = need
                st.rerun()
        return

    # 已提交：查询进度
    try:
        r = requests.get(f"{API_BASE}/video/status/{job_id}", timeout=30)
        if r.status_code == 404:
            m["video_status"] = "failed"
            m["video_error"] = "任务已过期（服务重启后任务不保留），请重新提交"
        else:
            r.raise_for_status()
            info = r.json()
            m["video_status"] = info.get("status", "unknown")
            m["video_prompts"] = info.get("storyboard", [])
            m["video_phase"] = info.get("phase", "")
            if m["video_status"] == "failed":
                m["video_error"] = info.get("error", "生成失败")
    except Exception as exc:
        m["video_status"] = "unknown"
        m["video_error"] = str(exc)

    status = m.get("video_status")
    if status == "succeed":
        try:
            r = requests.get(f"{API_BASE}/video/file/{job_id}", timeout=180)
            r.raise_for_status()
            video_bytes = r.content
        except Exception as exc:
            st.error(f"视频加载失败: {exc}")
            return
        st.video(video_bytes)
        if m.get("video_prompts"):
            st.caption("分镜：" + " ｜ ".join(f"镜头{i+1}：{p[:40]}…" if len(p) > 40 else f"镜头{i+1}：{p}"
                                            for i, p in enumerate(m["video_prompts"])))
        st.download_button("⬇️ 下载视频", data=video_bytes,
                           file_name=f"video_{job_id}.mp4", mime="video/mp4", key=f"vdl_{midx}")
    elif status in ("storyboard", "generating", "unknown"):
        prompts = m.get("video_prompts") or []
        detail = "\n\n".join(f"镜头{i+1}（已优化）：{p}" for i, p in enumerate(prompts)) if prompts else ""
        st.info(
            f"⏳ {m.get('video_need', '')}：{m.get('video_phase', '生成中……')}"
            + ("（7B 分镜完成，正在逐镜头生成视频）" if prompts else "（正在等待 7B 生成分镜…）")
            + f"\n\n{detail}\n\n页面每 10 秒自动刷新"
        )
        time.sleep(10)
        st.rerun()
    else:
        st.error(f"视频生成失败：{m.get('video_error', '未知错误')}")
        if st.button("重新填写需求", key=f"vretry_{midx}"):
            m["video_job"] = None
            st.rerun()


# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("会话")
    if st.button("➕ 新会话"):
        st.session_state.session_id = _new_session()
        st.session_state.uploads = []  # 数据文件随会话隔离，新会话从空开始
        st.session_state.pending_clarify = None
        st.rerun()
    st.caption(f"当前会话: {st.session_state.session_id}")

    st.divider()
    with st.expander("🎬 历史视频（刷新页面后可从这里找回）"):
        try:
            jobs = requests.get(f"{API_BASE}/video/list", timeout=10).json().get("jobs", [])
            done = [j for j in jobs if j.get("status") == "succeed" and j.get("file")]
            if not done:
                st.caption("暂无已完成的视频")
            else:
                labels = [f"{j['question'][:24]}（{j['job_id'][:6]}）" for j in done]
                pick = st.selectbox("选择视频", range(len(done)), format_func=lambda i: labels[i], key="hist_video")
                j = done[pick]
                if st.button("▶️ 播放", key="hist_play"):
                    r = requests.get(f"{API_BASE}/video/file/{j['job_id']}", timeout=180)
                    r.raise_for_status()
                    st.video(r.content)
                    st.download_button("⬇️ 下载", data=r.content,
                                       file_name=f"video_{j['job_id']}.mp4", mime="video/mp4",
                                       key="hist_dl")
        except Exception as exc:
            st.caption(f"历史视频加载失败: {exc}")

    st.divider()
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        st.caption(
            f"后端: {'🟢' if health.get('ok') else '🔴'} | "
            f"Ollama: {'🟢' if health.get('ollama_online') else '🔴'}"
        )
        if health.get("data_files"):
            st.caption("本会话数据文件: " + "、".join(Path(p).name for p in health["data_files"]))
    except Exception:
        st.caption("后端: 🔴 未连接（请先启动 uvicorn）")

# ---------- 对话区 ----------
if True:
    for i, m in enumerate(msgs):
        with st.chat_message(m["role"]):
            if m["role"] == "assistant" and m.get("intent") == "video":
                st.markdown(m["content"])
                render_video_card(m, i)
                st.caption(f"本次由 🎬 视频生成 处理")
                continue
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("images"):
                for img_b64 in m["images"]:
                    st.image(io.BytesIO(base64.b64decode(img_b64)), use_container_width=True)
            if m["role"] == "assistant" and m.get("code"):
                with st.expander("查看生成的分析代码"):
                    st.code(m["code"], language="python")
                if m.get("stdout"):
                    with st.expander("查看执行输出"):
                        st.code(m["stdout"], language="text")
            if m["role"] == "assistant" and m.get("intent"):
                label = INTENT_LABELS.get(m["intent"], m["intent"])
                st.caption(f"本次由 {label} 处理（置信度 {m.get('confidence', 0):.2f}）")

    question = st.chat_input(
        "例如：退租结算流程是什么 / 分析一下账单欠费情况 / 帮我推广一下新房源",
        accept_file="multiple",
        file_type=["pdf", "docx", "xlsx", "csv", "txt"],
    )

    # accept_file=True 时返回 ChatInputValue（.text + .files），不是 str/UploadedFile
    text, files = "", []
    if isinstance(question, str):
        text = question
    elif question is not None:
        text = question.text or ""
        files = list(question.files or [])

    # 附件：Excel 只登记为分析数据，其余入库
    for f in files:
        ext = Path(f.name).suffix.lower()
        is_excel = ext in EXCEL_EXTS
        kind = "data" if is_excel else "kb"
        try:
            resp = requests.post(
                f"{API_BASE}/upload",
                files={"file": (f.name, f.getvalue(), f.type or "application/octet-stream")},
                data={"kind": kind, "session_id": sid},
                timeout=600,
            )
            resp.raise_for_status()
            info = resp.json()
            tag = "登记为分析数据（Excel 不进知识库）" if is_excel else f"已入知识库（{info['ingested_chunks']} 块）"
            note = f"📎 上传附件 **{f.name}**：{tag}"
        except Exception as exc:
            note = f"📎 上传附件 **{f.name}** 失败: {exc}"
        msgs.append({"role": "user", "content": note})

    if text.strip():
        question = text
        msgs.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.info("思考中……")
            try:
                history = [m["content"][:200] for m in msgs[-6:] if m.get("content")]
                resp = requests.post(
                    f"{API_BASE}/chat",
                    json={"question": question, "session_id": sid, "history": history},
                    timeout=900,
                )
                resp.raise_for_status()
                result = resp.json()
            except Exception as exc:
                result = {"intent": "error", "answer": f"请求后端失败: {exc}", "images": []}

            placeholder.empty()
            st.markdown(result.get("answer", ""))
            intent = result.get("intent", "unknown")
            st.caption(f"本次由 {INTENT_LABELS.get(intent, intent)} 处理（置信度 {result.get('confidence', 0):.2f}）")
            render_result(result)

        msg: dict = {
            "role": "assistant",
            "content": result.get("answer", ""),
            "intent": intent,
            "confidence": result.get("confidence", 0.0),
            "images": result.get("images", []),
        }
        if intent == "video":
            msg["video_job"] = None  # 由下方卡片交互提交
            msg["video_origin"] = question
        if intent in ("clarify_analysis", "clarify_route") and result.get("clarify_id"):
            st.session_state.pending_clarify = {
                "clarify_id": result["clarify_id"],
                "questions": result.get("questions", []),
                "endpoint": "/analysis/answer" if intent == "clarify_analysis" else "/clarify/answer",
            }
        msgs.append(msg)

        # 视频意图：立即渲染卡片（重跑后由渲染循环接管）
        if intent == "video":
            st.rerun()

    # ---------- 澄清交互 ----------
    pc = st.session_state.pending_clarify
    if pc:
        st.divider()
        header = "🔍 需求确认" if pc.get("endpoint") == "/analysis/answer" else "🔀 选择回答方式"
        st.subheader(header)
        answers: dict[str, str] = {}
        for q in pc.get("questions", []):
            qid = q.get("id", "q")
            st.markdown(f"**{q.get('question', '')}**")
            opts = list(q.get("options") or [])
            if not opts:
                opts = ["继续按当前理解执行", "其他"]
            choice = st.radio(
                qid, opts,
                key=f"clarify_{pc['clarify_id']}_{qid}",
                label_visibility="collapsed",
            )
            answers[qid] = choice
            if choice == "其他":
                custom = st.text_input(
                    "请补充说明", key=f"clarify_custom_{pc['clarify_id']}_{qid}",
                    placeholder="写清具体口径，例如：确权表'房源'末尾4位数字 = 房间列表'房间名称'；面积用'建筑面积(㎡)'列；不排除退租",
                )
                if custom.strip():
                    answers[qid] = custom

        submit_disabled = st.session_state.clarify_submitting
        if submit_disabled:
            st.info("⏳ 正在执行中，请勿重复提交……")
        if st.button("✅ 提交", disabled=submit_disabled):
            st.session_state.clarify_submitting = True
            try:
                with st.spinner("正在处理，可能需要几分钟……"):
                    try:
                        resp = requests.post(
                            f"{API_BASE}{pc.get('endpoint', '/analysis/answer')}",
                            json={"clarify_id": pc["clarify_id"], "answers": answers},
                            timeout=900,
                        )
                        resp.raise_for_status()
                        result = resp.json()
                    except requests.HTTPError as exc:
                        if exc.response is not None and exc.response.status_code == 404:
                            result = {"intent": "info", "answer": "", "images": []}
                        else:
                            result = {"intent": "error", "answer": f"请求后端失败: {exc}", "images": []}
                    except Exception as exc:
                        result = {"intent": "error", "answer": f"请求后端失败: {exc}", "images": []}
            finally:
                st.session_state.clarify_submitting = False
            st.session_state.pending_clarify = None
            if result.get("intent") != "info":
                msgs.append({
                    "role": "user",
                    "content": "（已提交选择）" + "；".join(str(v) for v in answers.values()),
                })
                intent = result.get("intent", "unknown")
                with st.chat_message("assistant"):
                    st.markdown(result.get("answer", ""))
                    st.caption(f"本次由 {INTENT_LABELS.get(intent, intent)} 处理（置信度 {result.get('confidence', 0):.2f}）")
                    render_result(result)
                msg = {
                    "role": "assistant",
                    "content": result.get("answer", ""),
                    "intent": intent,
                    "confidence": result.get("confidence", 0.0),
                    "images": result.get("images", []),
                }
                if intent == "video":
                    msg["video_job"] = None
                msgs.append(msg)
            st.rerun()
