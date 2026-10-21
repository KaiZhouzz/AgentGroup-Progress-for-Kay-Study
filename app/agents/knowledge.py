"""RAG 知识库 agent：文件入库（Chroma）+ 检索 + 带引用回答。纯本地可跑。"""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from app.core import model_router
from app.core.config import settings
from app.parsers import load_document

logger = logging.getLogger(__name__)

COLLECTION = "knowledge"

_vstore: Chroma | None = None


def _get_store() -> Chroma:
    global _vstore
    if _vstore is None:
        if settings.chroma_http_url.strip():
            # 远程 Chroma 服务：知识库统一存储到服务端，本机不再落盘向量
            from chromadb import HttpClient

            from urllib.parse import urlparse

            u = urlparse(settings.chroma_http_url.strip())
            client = HttpClient(host=u.hostname or "localhost", port=u.port or 8000)
            logger.info("using remote chroma at %s", settings.chroma_http_url)
            _vstore = Chroma(
                collection_name=COLLECTION,
                embedding_function=model_router.embeddings(),
                client=client,
            )
        else:
            settings.chroma_path.mkdir(parents=True, exist_ok=True)
            _vstore = Chroma(
                collection_name=COLLECTION,
                embedding_function=model_router.embeddings(),
                persist_directory=str(settings.chroma_path),
            )
    return _vstore


def ingest_file(path: str | Path, dept: str = "") -> int:
    """解析文件、切块、嵌入并写入 Chroma，返回入库 chunk 数。

    同一文件重复上传时覆盖旧内容：先按 source 删除旧分块再写入，避免知识库堆积重复向量。
    dept 为部门权限预留字段：仅写入 metadata，检索侧暂不过滤（见 search_knowledge）。
    """
    path = Path(path)
    chunks = load_document(path)
    if not chunks:
        return 0
    docs = [
        Document(page_content=c, metadata={"source": path.name, "department": dept})
        for c in chunks
    ]
    store = _get_store()
    removed = store._collection.delete(where={"source": path.name})
    store.add_documents(docs)
    if not settings.chroma_http_url.strip():
        store.persist()  # 远程模式下服务端自动持久化，无需显式调用
    logger.info("ingested %s (%d chunks, replaced previous)", path.name, len(docs))
    return len(docs)


def list_sources() -> list[dict]:
    """按来源文件聚合的入库清单：source / 切片数 / 部门。"""
    res = _get_store()._collection.get(include=["metadatas"])
    agg: dict[str, dict] = {}
    for m in res["metadatas"]:
        src = m.get("source")
        if not src:
            continue
        e = agg.setdefault(src, {"source": src, "chunks": 0, "department": m.get("department", "")})
        e["chunks"] += 1
    return sorted(agg.values(), key=lambda x: x["source"])


def list_chunks(source: str) -> list[dict]:
    """某个文件的全部切片：id / 前 120 字预览 / 部门。"""
    res = _get_store()._collection.get(where={"source": source}, include=["metadatas", "documents"])
    return [
        {"id": i, "preview": (d or "").replace("\n", " ")[:120], "department": m.get("department", "")}
        for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])
    ]


def delete_source(source: str) -> int:
    """按来源文件删除全部切片，返回删除的切片数。"""
    store = _get_store()
    res = store._collection.get(where={"source": source})
    ids = res.get("ids") or []
    if ids:
        store._collection.delete(ids=ids)
        if not settings.chroma_http_url.strip():
            store.persist()
    logger.info("deleted source %s (%d chunks)", source, len(ids))
    return len(ids)


# 问题里的常见疑问/量词/单位，不作为关键词直通条件
_STOPWORDS = {
    "多少", "是什么", "什么是", "什么", "怎么", "如何", "为什么", "哪里", "哪些",
    "的", "了", "和", "与", "及", "吗", "呢", "是", "在", "有", "为",
    "用量", "数量", "价格", "单价", "金额", "总和", "总计", "合计", "流程", "规定", "制度",
    # 计量单位：几乎每行都有，命中无区分度
    "公斤", "千克", "斤", "克", "吨", "升", "毫升", "元", "块", "个", "件", "箱", "瓶",
    "袋", "包", "只", "条", "份", "次", "天", "月", "年", "平方米", "平米",
    # 常见动词
    "用了", "用掉", "花了", "使用",
}


def _extract_tokens(query: str) -> list[str]:
    try:
        import jieba
    except ImportError:
        return []
    return sorted(
        {t.strip() for t in jieba.lcut(query) if len(t.strip()) >= 2 and t.strip() not in _STOPWORDS and not t.strip().isdigit()},
        key=len,
        reverse=True,
    )


def _keyword_hits(tokens: list[str], limit: int) -> list[Document]:
    """关键词直通：问题中切出的实词若在文档里出现（忽略大小写），直接命中该文档。

    纯向量相似度对"表格里找某个具体品名"这类短词查询效果差（embedding 把
    白菜/白芝麻/白猫洗洁精混为一谈），用精确匹配补足。

    不用 Chroma 的 $contains（大小写敏感，且 iOS/IOS 这类混写变体覆盖不全）：
    全库取回后在 Python 侧做 case-insensitive 匹配，排名按"命中词数"计分——
    多词同时命中的文档（如"阿里云短信接口"+"账号"同现的那一行）优先，
    避免高频词（账号/名称）按库顺序把名额占满、真正的目标行被挤出。
    """
    if not tokens:
        return []

    try:
        res = _get_store()._collection.get(include=["documents", "metadatas"])
    except Exception:
        return []
    lowered = [t.lower() for t in tokens]
    scored: list[tuple[int, str, dict]] = []
    for _doc_id, content, meta in zip(res["ids"], res["documents"], res["metadatas"]):
        c = (content or "").lower()
        s = sum(1 for t in lowered if t in c)
        if s:
            scored.append((s, content, meta))
    # 命中词多的排前面；同分保持库内顺序（stable sort）
    scored.sort(key=lambda x: -x[0])
    return [Document(page_content=c, metadata=m) for _, c, m in scored[:limit]]


def _snippet(doc: Document, tokens: list[str], window: int = 0, max_lines: int = 12) -> Document:
    """从命中文档里抽取"标题 + 含关键词的行"的片段，而不是整页。

    整页表格动辄两三千字，塞进 prompt 既拖慢生成又干扰阅读。
    行已是"品名=xx，数量=xx"的自描述格式，无需邻居行补上下文——
    留邻居反而会让小模型抄错行（把相邻品名的数值当答案）。
    行抽取按"命中词数"打分：多词同现的行（如"名称=阿里云短信接口; 账号=…"）优先保留，
    避免每行都含的泛词（账号/名称）把目标行挤出 max_lines。
    """
    lines = doc.page_content.split("\n")
    if not tokens:
        return Document(page_content="\n".join(lines[:max_lines]), metadata=doc.metadata)

    lowered = [t.lower() for t in tokens]
    low_lines = [line.lower() for line in lines]
    score = [sum(1 for t in lowered if t in low_lines[i]) for i in range(len(lines))]
    keep = {0}  # 页首标题（清单名）始终保留，供回答注明出处
    keep.update(i for i, s in enumerate(score) if s > 0)
    if len(keep) > max_lines:
        # 标题 + 按命中词数降序截取（同分保持行序），超出行丢弃
        rest = sorted((i for i in keep if i != 0), key=lambda i: (-score[i], i))[: max_lines - 1]
        keep = {0, *rest}
    picked = [lines[i] for i in sorted(keep)]
    return Document(page_content="\n".join(picked), metadata=doc.metadata)


def search_knowledge(query: str, k: int = 4) -> list[Document]:
    store = _get_store()
    tokens = _extract_tokens(query)
    # 关键词命中放宽到 2k：$contains 是子串匹配（"白菜"也会命中"杭白菜"），
    # 放宽名额避免真正的目标页被相近词页面挤掉，取舍交给生成模型按 prompt 规则处理
    hits = [_snippet(d, tokens) for d in _keyword_hits(tokens, limit=k * 2)]
    if not hits:
        return store.similarity_search(query, k=k)
    # 去重按"页首标题"（hits 是片段，全文对比会失效），
    # 向量补充只保留确实含关键词的页，并同样片段化，避免整页噪音回灌
    seen = {d.page_content.split("\n", 1)[0] for d in hits}
    fill: list[Document] = []
    for d in store.similarity_search(query, k=k * 2):
        head = d.page_content.split("\n", 1)[0]
        if head in seen:
            continue
        seen.add(head)
        if tokens and not any(t in d.page_content for t in tokens):
            continue
        fill.append(_snippet(d, tokens))
    return hits + fill


RAG_PROMPT = """你是长租公寓运营的客服助手。请严格根据以下知识库片段回答问题。
如果片段中没有相关信息，请明说"知识库中未找到相关内容"，不要编造。

【知识库片段】
{context}

【问题】
{question}

回答要求：
1. 用中文，条理清晰，先结论后细节。
2. 回答末尾必须列出引用的文件名，格式：引用：<文件名>（可多个）。
3. 如果同一指标在不同清单/时期/地点的取值不同，必须逐一列出每个取值，并注明各自
   对应的清单（页首标题），禁止只报一个值或擅自合并；片段信息不足时如实说明。
4. 片段中每行是一条独立记录；回答某个品名/对象的数值时，只取"品名规格=该品名"
   那一行里的对应字段，严禁混入相邻行的数值（相邻行是别的品名，不是答案）。
"""


def answer(query: str, k: int = 4) -> dict:
    docs = search_knowledge(query, k=k)
    context = "\n\n---\n\n".join(
        f"[来源: {d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs
    )
    llm = model_router.chat_model_for("rag", temperature=0.2)
    prompt = RAG_PROMPT.format(context=context or "（知识库为空）", question=query)
    text = llm.invoke(prompt).content
    sources = sorted({d.metadata.get("source", "?") for d in docs})
    return {"answer": text, "sources": sources, "intent": "knowledge"}
