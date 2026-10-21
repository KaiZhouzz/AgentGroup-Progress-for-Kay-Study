"""LangGraph 意图路由：classify（规则优先，LLM 兜底）→ knowledge / analysis / aigc。"""
from __future__ import annotations

import logging
import re
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.agents import aigc, analysis, knowledge
from app.core import model_router

logger = logging.getLogger(__name__)

Intent = Literal["knowledge", "analysis", "aigc", "video", "forecast", "clarify", "clarify_route"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    question: str
    data_files: list[str]      # 已上传的数据文件绝对路径
    kb_sources: list[str]      # 已入库知识库文件名
    history: list[str]         # 最近对话文本，供分类器理解追问
    force_intent: str          # 用户通过路由追问选定后强制执行的意图
    clarify_kind: str          # clarify_route 的追问类型："aigc"=文案/视频，缺省=知识库/数据
    intent: Intent
    confidence: float
    result: dict[str, Any]


# 规则层关键词：命中即走规则分类，置信度 0.9
_ANALYSIS_PAT = re.compile(
    r"分析|统计|出租率|收缴率|欠费|账单|营收|收入|退租率|入住率|环比|同比|汇总|排名|趋势|计算"
    r"|脚本|模型|公式|预测|建模|回归|坪效|评估|测算"
)
_AIGC_PAT = re.compile(r"文案|招租|推广|海报|标题|朋友圈|社群话术|活动通知|公告|宣传")
_VIDEO_PAT = re.compile(r"视频|宣传片|短片|动画|影片")
# 明确是文字创作的词（命中即 aigc，不反问）
_COPY_PAT = re.compile(r"文案|标题|朋友圈|话术|通知|公告|标语|口号|软文|脚本")
# 创作类但载体不明（文案还是视频，反问用户）
_AMBIG_AIGC_PAT = re.compile(r"招租|推广|宣传|海报|营销")
# 预测分析（预留口子）：先占位应答，真正的预测 agent 后续接入 forecast 节点
_FORECAST_PAT = re.compile(r"预测|预估|预警|走势|将来|未来.*(趋势|需求|租金|空置)|forecast", re.S)
_KB_HINT_PAT = re.compile(r"流程|制度|规定|怎么|如何|是什么|怎么办|政策|合同|押金|退租|维修|保洁|水电")

CLASSIFY_PROMPT = """你是长租公寓运营平台的意图分类器。把用户问题分到三类之一：

- knowledge: 询问制度、流程、政策、规定类知识（如退租流程、水电费规定）。
- analysis: 要求对数据做统计、分析、计算、可视化（通常涉及账单、出租率、收缴率等指标）。
- aigc: 要求生成文案、招租推广、活动通知等创作内容。

只输出 JSON，格式：{{"intent": "...", "confidence": 0.0-1.0}}
不要输出其他内容。

【最近对话】（帮助理解追问的指代和语境，可能为空）
{history}

用户问题：{question}

注意：如果当前问题是追问（含"他们/这些/上面/具体/继续"等指代），要结合最近对话判断它延续的是哪类意图。"""

SOURCE_PROMPT = """用户在长租公寓运营平台提问。这个问题既可能由【知识库文档】回答（RAG 检索文档片段），
也可能由【数据文件】回答（写代码精确统计/过滤）。请判断哪个数据源更合适。

【知识库文件】
{kb_sources}

【数据文件】
{data_files}

【最近对话】
{history}

【用户问题】
{question}

判断标准：
- knowledge：问题是文档型点查（制度、流程、某条记录的具体描述），或只需要看几条记录、不追求全量精确。
- analysis：问题涉及计算、汇总、排名、全量列出、精确数字过滤（这类必须跑代码才能保证完整准确）。

只输出 JSON：{{"source": "knowledge" 或 "analysis", "confident": true 或 false}}
confident=false 表示两种方式各有利弊、无法明确判断。"""


def _rule_classify(question: str, state: AgentState) -> tuple[Intent, float]:
    """规则层：数据文件优先 + 关键词匹配。返回 (intent, confidence)。"""
    has_data = bool(state.get("data_files"))
    if _ANALYSIS_PAT.search(question) and (has_data or "分析" in question or "统计" in question):
        return "analysis", 0.9
    if _AIGC_PAT.search(question):
        return "aigc", 0.9
    if _KB_HINT_PAT.search(question):
        return "knowledge", 0.8
    if has_data:
        # 有数据文件但问题含糊，倾向分析
        return "analysis", 0.6
    return "knowledge", 0.5  # 默认走知识库


def classify_node(state: AgentState) -> AgentState:
    question = state["question"]
    if state.get("force_intent"):
        # 用户在路由追问中已选定方式，跳过分歧直接执行
        return {"intent": state["force_intent"], "confidence": 0.95}

    intent, conf = _rule_classify(question, state)

    # AIGC 二级判断：文案还是视频。"视频文案/视频脚本"仍是文字创作归 aigc；
    # 明确视频词 → video；只有"推广/宣传"类模糊词 → 反问用户选文案还是视频
    if "视频文案" not in question and "视频脚本" not in question and _VIDEO_PAT.search(question):
        intent, conf = "video", 0.9
    elif _COPY_PAT.search(question):
        intent, conf = "aigc", 0.9
    elif _AMBIG_AIGC_PAT.search(question):
        logger.info("创作类需求载体不明，追问文案/视频: %r", question[:50])
        return {"intent": "clarify_route", "confidence": 0.5, "clarify_kind": "aigc"}
    elif _FORECAST_PAT.search(question):
        # 预测意图预留口子：规则命中即占位应答；后续在此接入真正的预测分析 agent
        intent, conf = "forecast", 0.85

    # 规则置信度低时用 LLM 再分一次（带最近对话，理解追问语境）
    if conf < 0.7:
        try:
            llm = model_router.chat_model_for("chat", temperature=0.0)
            import json as _json

            history = "\n".join(f"- {h}" for h in (state.get("history") or [])[-6:] if h.strip())
            raw = llm.invoke(CLASSIFY_PROMPT.format(
                question=question, history=history or "（无）",
            )).content
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                parsed = _json.loads(m.group(0))
                llm_intent = parsed.get("intent", "knowledge")
                llm_conf = float(parsed.get("confidence", 0.5))
                if llm_conf >= conf:
                    intent, conf = llm_intent, llm_conf
        except Exception as exc:
            logger.warning("LLM 分类失败，沿用规则结果: %s", exc)

    # 歧义追问：知识库和数据文件双通，且分类非规则关键词高置信命中 → 让语义模型判断数据源，
    # 判不了就反问用户选哪种方式（7B 的 confidence 校准差，不能只看分数）
    if (intent in ("knowledge", "analysis") and conf < 0.9
            and state.get("data_files") and state.get("kb_sources")):
        routed = _route_by_source(state)
        if routed == "ambiguous":
            logger.info("双链路歧义，追问用户选择: %r", question[:50])
            return {"intent": "clarify_route", "confidence": 0.5}
        if routed:
            intent, conf = routed, 0.88

    logger.info("intent=%s confidence=%.2f question=%r", intent, conf, question[:50])

    # 兜底：置信度太低则反问用户
    if conf < 0.45:
        intent = "clarify"
    return {"intent": intent, "confidence": conf}


def _route_by_source(state: AgentState) -> str | None:
    """双链路都通时，让语义模型判断数据源。返回 "knowledge"/"analysis"/None(判不了)/"ambiguous"。"""
    import json as _json

    llm = model_router.chat_model_for("chat", temperature=0.0)
    history = "\n".join(f"- {h}" for h in (state.get("history") or [])[-6:] if h.strip())
    prompt = SOURCE_PROMPT.format(
        question=state["question"],
        kb_sources="\n".join(f"- {s}" for s in (state.get("kb_sources") or [])) or "（无）",
        data_files="\n".join(f"- {s.split('/')[-1]}" for s in (state.get("data_files") or [])) or "（无）",
        history=history or "（无）",
    )
    try:
        raw = llm.invoke(prompt).content
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        parsed = _json.loads(m.group(0))
        if not parsed.get("confident", False):
            return "ambiguous"
        src = parsed.get("source")
        return src if src in ("knowledge", "analysis") else None
    except Exception as exc:
        logger.warning("数据源判断失败: %s", exc)
        return None


def _clarify(state: AgentState) -> AgentState:
    answer_text = (
        "我不太确定您的需求属于哪一类。您可以：\n"
        "1. 咨询运营制度/流程（如'退租结算流程是什么'）\n"
        "2. 上传账单等数据后让我做分析（如'分析一下账单欠费情况'）\n"
        "3. 让我帮忙写运营文案（如'写一条招租文案'）\n"
        "请换个说法或补充说明～"
    )
    return {"result": {"intent": "clarify", "answer": answer_text}}


def _route_clarify(state: AgentState) -> AgentState:
    """双链路歧义追问：让用户选回答方式，选择后强制走对应链路。

    clarify_kind="aigc"：创作类需求载体不明，选文案还是视频；
    否则（默认）：知识库文档 vs 数据文件两种回答方式。
    """
    import time
    import uuid

    if state.get("clarify_kind") == "aigc":
        return {"result": {
            "intent": "clarify_route",
            "answer": "好的，您想做哪种形式的创作？",
            "questions": [{
                "id": "route",
                "question": "选择创作形式：",
                "options": ["✍️ 写文案（招租文案/标题/通知等文字内容）", "🎬 做视频（宣传短片，每镜头 5 秒）"],
            }],
            "session": {
                "session_id": uuid.uuid4().hex[:12],
                "question": state["question"],
                "clarify_kind": "aigc",
                "updated_at": time.time(),
            },
            "success": True, "images": [], "code": "", "stdout": "", "stderr": "",
        }}
    return {"result": {
        "intent": "clarify_route",
        "answer": "这个问题从知识库文档和数据文件两种方式都能回答，请选一种：",
        "questions": [{
            "id": "route",
            "question": "你希望用哪种方式回答？",
            "options": [
                "从知识库文档查找（适合：制度/点查/快速回答）",
                "对数据文件精确统计（适合：计算/全量列出/精确数字）",
                "其他",
            ],
        }],
        "session": {
            "session_id": uuid.uuid4().hex[:12],
            "question": state["question"],
            "updated_at": time.time(),
        },
        "success": True, "images": [], "code": "", "stdout": "", "stderr": "",
    }}


def run_knowledge(state: AgentState) -> AgentState:
    return {"result": knowledge.answer(state["question"])}


def run_analysis(state: AgentState) -> AgentState:
    files = state.get("data_files", [])
    if not files:
        # 分析链路需要数据文件；若知识库有内容，数值点查类问题（如"大白菜用了多少公斤"）
        # 也可能由文档回答，回退知识库而不是直接拒绝
        if state.get("kb_sources"):
            logger.info("analysis 意图但无数据文件，回退知识库: %r", state["question"][:50])
            return {"result": knowledge.answer(state["question"])}
        return {"result": {
            "intent": "analysis",
            "answer": "您还没有上传数据文件。请先在左侧上传账单/运营数据（xlsx、csv），再提出分析需求。",
            "success": False, "images": [], "code": "", "stdout": "", "stderr": "",
        }}
    from app.agents.pipeline import AnalysisPipeline

    return {"result": AnalysisPipeline(files).start(state["question"])}


def run_aigc(state: AgentState) -> AgentState:
    return {"result": aigc.generate(state["question"])}


def run_video(state: AgentState) -> AgentState:
    """视频意图：不直接提交任务，引导前端在对话内渲染视频生成卡片。"""
    return {"result": {
        "intent": "video",
        "answer": (
            "好的，我来帮您制作宣传视频。请在下方卡片中：\n"
            "1. 描述您想要的画面（场景、镜头、氛围）\n"
            "2. 选择镜头数（每镜头 5 秒，多个镜头自动衔接拼接）\n"
            "3. 可选：上传一张参考图作为视频起始画面\n\n"
            "确认后点击提交，生成需要几分钟，请稍候。"
        ),
        "success": True, "images": [], "code": "", "stdout": "", "stderr": "",
    }}


def run_forecast(state: AgentState) -> AgentState:
    """预测分析意图（预留口子）：当前为占位应答，后续在此接入预测分析 agent。

    规划中的能力：基于历史运营数据（出租率/收缴率/退租记录）做时间序列预测、
    欠费风险预警、租金/需求走势估计。数据依赖对话附件上传的数据文件。
    """
    return {"result": {
        "intent": "forecast",
        "answer": (
            "您提到的是预测类需求（预测/预警/走势）。🚧 预测分析 agent 正在规划中，"
            "届时将支持：基于历史数据的出租率/收缴率预测、欠费风险预警、租金走势估计等。\n\n"
            "当前您可以把这类问题改述为统计分析（如'统计过去 6 个月的退租率变化'），"
            "数据分析 agent 可以先帮您把历史规律摸清楚。"
        ),
        "success": True, "images": [], "code": "", "stdout": "", "stderr": "",
    }}


def route_after_classify(state: AgentState) -> str:
    return state.get("intent", "knowledge")


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("classify", classify_node)
    g.add_node("knowledge", run_knowledge)
    g.add_node("analysis", run_analysis)
    g.add_node("aigc", run_aigc)
    g.add_node("video", run_video)
    g.add_node("forecast", run_forecast)
    g.add_node("clarify", _clarify)
    g.add_node("route_clarify", _route_clarify)

    g.set_entry_point("classify")
    g.add_conditional_edges("classify", route_after_classify, {
        "knowledge": "knowledge",
        "analysis": "analysis",
        "aigc": "aigc",
        "video": "video",
        "forecast": "forecast",
        "clarify": "clarify",
        "clarify_route": "route_clarify",
    })
    for node in ("knowledge", "analysis", "aigc", "video", "forecast", "clarify", "route_clarify"):
        g.add_edge(node, END)
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def handle_question(question: str, data_files: list[str] | None = None, kb_sources: list[str] | None = None,
                    history: list[str] | None = None, force_intent: str | None = None) -> dict:
    """统一入口：跑一遍图，返回结果 dict（含 intent/answer 等）。"""
    graph = get_graph()
    state: AgentState = {
        "messages": [("user", question)],
        "question": question,
        "data_files": data_files or [],
        "kb_sources": kb_sources or [],
        "history": history or [],
    }
    if force_intent:
        state["force_intent"] = force_intent
    final = graph.invoke(state)
    result = final.get("result") or {}
    result.setdefault("intent", final.get("intent", "knowledge"))
    result["confidence"] = final.get("confidence", 0.0)
    return result
