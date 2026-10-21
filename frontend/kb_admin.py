"""知识库管理页面（独立页，通过侧边栏导航进入）。

功能：登录（预留 admin/admin）→ 查看已入库文件及切片详情 → 删除文件。
注意：登录是前端占位校验，真正的权限控制需要后端鉴权（后续版本）。
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE = os.environ.get("KT_API_BASE", "http://localhost:8000")
ADMIN_USER = "admin"
ADMIN_PASS = "admin"  # 预留账号，正式版改后端鉴权 + 环境变量

st.set_page_config(page_title="知识库管理 - 长租公寓运营 Agent 平台", page_icon="📚")
st.title("📚 知识库管理")

if "kb_admin" not in st.session_state:
    st.session_state.kb_admin = False

# ---------- 登录 ----------
if not st.session_state.kb_admin:
    st.caption("知识库管理需要登录（测试账号：admin / admin）")
    with st.form("kb_login"):
        user = st.text_input("账号")
        pwd = st.text_input("密码", type="password")
        if st.form_submit_button("登录"):
            if user == ADMIN_USER and pwd == ADMIN_PASS:
                st.session_state.kb_admin = True
                st.rerun()
            else:
                st.error("账号或密码错误")
    st.stop()


def api_get(path: str) -> dict:
    r = requests.get(f"{API_BASE}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


top1, top2 = st.columns([4, 1])
top1.caption("已登录：admin ｜ 删除操作不可恢复，请谨慎执行")
if top2.button("退出登录"):
    st.session_state.kb_admin = False
    st.rerun()

# ---------- 上传 ----------
st.divider()
st.subheader("⬆️ 上传入库")
st.caption("支持 pdf / docx / txt / xlsx / csv；Excel 从本页上传时入知识库（对话附件中的 Excel 仍登记为分析数据）")
DEPT_OPTIONS = ["不限", "运营部", "客服部", "工程部", "财务部", "人事部"]
dept = st.selectbox("所属部门（权限预留）", DEPT_OPTIONS, key="dept_select")
up = st.file_uploader(
    "选择文件（可多选）",
    type=["pdf", "docx", "txt", "xlsx", "xls", "csv"],
    accept_multiple_files=True,
    key="kb_upload",
)
if up and st.button(f"上传 {len(up)} 个文件入库", key="do_upload"):
    flashes = []
    for f in up:
        try:
            resp = requests.post(
                f"{API_BASE}/upload",
                files={"file": (f.name, f.getvalue(), f.type or "application/octet-stream")},
                data={"kind": "kb", "dept": dept},
                timeout=600,
            )
            resp.raise_for_status()
            info = resp.json()
            if info["ingested_chunks"] == 0:
                flashes.append(("warning", f"{info['filename']}：未提取到文本，未入库"))
            else:
                flashes.append(("success", f"{info['filename']}：入库 {info['ingested_chunks']} 块（部门：{dept}）"))
        except Exception as exc:
            flashes.append(("error", f"{f.name} 上传失败: {exc}"))
    st.session_state.upload_flash = flashes
    st.rerun()

# rerun 后展示上次上传结果（直接在 button 分支里 st.success 会被 rerun 瞬间清掉）
for kind, msg in st.session_state.pop("upload_flash", []):
    getattr(st, kind)(msg)

# ---------- 文件列表 + 删除 ----------
try:
    sources = api_get("/kb/sources")["sources"]
except Exception as exc:
    st.error(f"无法连接后端: {exc}")
    st.stop()

st.subheader(f"已入库文件（{len(sources)} 个）")
if not sources:
    st.info("知识库为空。可在本页上传，或在对话中通过附件上传 PDF/DOCX/TXT。")

for s in sources:
    with st.container(border=True):
        c1, c2, c3 = st.columns([4, 2, 1])
        c1.markdown(f"**📄 {s['source']}**")
        c2.caption(f"{s['chunks']} 个切片 ｜ 部门：{s.get('department') or '不限'}")
        if c3.button("删除", key=f"del_{s['source']}", type="primary"):
            st.session_state[f"confirm_{s['source']}"] = True

        if st.session_state.get(f"confirm_{s['source']}"):
            st.warning(f"确认删除 **{s['source']}** 的全部 {s['chunks']} 个切片？此操作不可恢复。")
            ok, cancel = st.columns(2)
            if ok.button("确认删除", key=f"ok_{s['source']}"):
                try:
                    r = requests.delete(f"{API_BASE}/kb/source", params={"source": s["source"]}, timeout=60)
                    r.raise_for_status()
                    st.success(f"已删除 {r.json()['removed_chunks']} 个切片")
                    del st.session_state[f"confirm_{s['source']}"]
                    st.rerun()
                except Exception as exc:
                    st.error(f"删除失败: {exc}")
            if cancel.button("取消", key=f"cancel_{s['source']}"):
                del st.session_state[f"confirm_{s['source']}"]
                st.rerun()

# ---------- 切片详情 ----------
st.divider()
st.subheader("🔍 切片详情")
if sources:
    pick = st.selectbox("选择文件", [s["source"] for s in sources])
    if st.button("查看切片", key="view_chunks"):
        try:
            chunks = api_get(f"/kb/chunks?source={requests.utils.quote(pick)}")["chunks"]
            st.caption(f"共 {len(chunks)} 个切片，预览前 120 字")
            for i, ch in enumerate(chunks, 1):
                with st.expander(f"切片 {i} ｜ 部门：{ch.get('department') or '不限'}"):
                    st.caption(ch["id"])
                    st.write(ch["preview"])
        except Exception as exc:
            st.error(f"查询失败: {exc}")
