"""模型路由：本地 Ollama 优先，可选阿里云百炼 DashScope 用于复杂任务。

无 DASHSCOPE_API_KEY 时所有能力降级到本地 qwen2.5：
- 对话 / RAG / 文案生成：本地即可，效果可接受。
- 数据分析代码生成：本地模型可用，但成功率低于云端模型，
  通过 prompt 内 few-shot 示例（见 agents/analysis.py）兜底。
"""
from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.core.config import settings

logger = logging.getLogger(__name__)


def local_chat_model(temperature: float = 0.3, num_ctx: int = 16384) -> BaseChatModel:
    """本地 Ollama 对话模型，永远可用（只要 Ollama 服务在线）。

    num_ctx 默认 16k：分析场景的 prompt（数据概要+结构探查+规则）可能很长，
    4k/8k 会静默截断前缀导致模型看不到关键指令。16k 在 16G 内存的 Mac 上可承受。
    """
    return ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        num_ctx=num_ctx,
    )


def cloud_chat_model(temperature: float = 0.3) -> BaseChatModel | None:
    """云端 DashScope 对话模型；未配置 Key 时返回 None。

    DashScope 兼容 OpenAI 接口，用 langchain-openai 接入。
    """
    if not settings.has_cloud_key:
        return None
    from langchain_openai import ChatOpenAI

    logger.info("使用云端模型 %s（DashScope）", settings.dashscope_model)
    return ChatOpenAI(
        model=settings.dashscope_model,
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        temperature=temperature,
    )


def _ollama_model_available(name: str) -> bool:
    """探测 Ollama 本地是否已拉取某个模型。

    注意：必须用禁用代理的 opener——macOS 上 urllib 会读系统代理，
    代理节点波动时连 localhost 都会返回 502，导致误判模型不可用。
    """
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{settings.ollama_base_url}/api/tags", timeout=3) as resp:
            import json

            names = {m.get("name", "") for m in json.load(resp).get("models", [])}
            if ":" in name:
                return name in names  # 带 tag 必须精确匹配（14b 不能拿 7b 冒充）
            return any(n.split(":")[0] == name for n in names)
    except Exception:
        return False


def local_planner_model(temperature: float = 0.2) -> BaseChatModel:
    """本地语义理解模型：优先 planner_model（默认通用 14B），未拉取回退 chat_model。

    用于需求质疑/澄清、spec 合并、报错裁判这类推理任务——识别口径歧义和
    提出好追问需要强指令遵循能力，7B 容易"硬答"不问，故独立路由。
    """
    model = settings.planner_model if _ollama_model_available(settings.planner_model) else settings.chat_model
    if model != settings.planner_model:
        logger.warning("本地未找到 %s，语义理解降级到 %s", settings.planner_model, model)
    return ChatOllama(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        num_ctx=16384,
    )


def local_code_model(temperature: float = 0.1) -> BaseChatModel:
    """本地代码生成模型：优先 code_model（默认 14b），未拉取依次回退 coder:7b、通用 chat 模型。"""
    candidates = [settings.code_model, "qwen2.5-coder:7b", settings.chat_model]
    model = next((c for c in candidates if _ollama_model_available(c)), settings.chat_model)
    if model != candidates[0]:
        logger.warning("本地未找到 %s，代码生成回退到 %s", settings.code_model, model)
    return ChatOllama(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        num_ctx=16384,
    )


def chat_model_for(task: str, temperature: float = 0.3) -> BaseChatModel:
    """按任务类型选择模型：complex 任务（代码生成）优先云端，失败/无 Key 回退本地。

    task: "chat" | "rag" | "code" | "copywriting"
    """
    if task == "code" and settings.has_cloud_key:
        cloud = cloud_chat_model(temperature=temperature)
        if cloud is not None:
            return cloud
        logger.warning("云端模型不可用，代码生成降级到本地 %s", settings.code_model)
        return local_code_model(temperature=temperature)
    if task == "rag":
        # 知识库回答：检索侧已做关键词精确匹配 + 键值对结构化（见 agents/knowledge.py），
        # 7B 足以可靠抽取；上下文片段化后生成很快，不再需要用 14B 换质量
        return local_chat_model(temperature=temperature)
    return local_chat_model(temperature=temperature)


def embeddings() -> OllamaEmbeddings:
    """本地 embedding 模型（nomic-embed-text），RAG 专用，无云端方案。"""
    return OllamaEmbeddings(
        model=settings.embed_model,
        base_url=settings.ollama_base_url,
    )


def ollama_available() -> bool:
    """探测 Ollama 服务是否可达（禁用代理，原因同 _ollama_model_available）。"""
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{settings.ollama_base_url}/api/tags", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False
