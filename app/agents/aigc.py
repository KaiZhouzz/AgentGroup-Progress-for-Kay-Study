"""AIGC 文案生成 agent（stub 级）：招租文案、社区活动文案等，本地模型即可。"""
from __future__ import annotations

import logging

from app.core import model_router

logger = logging.getLogger(__name__)

COPY_PROMPT = """你是长租公寓的运营文案专家。请根据需求生成一段中文运营文案。

【需求】
{question}

硬性规则（必须严格遵守）：
1. 只能使用【需求】中明确给出的信息。需求里没有的信息——社区名称、位置、价格、
   房型、设施、优惠力度、联系方式——一律用占位符【】标出（如【社区名称】），
   绝对禁止编造听起来合理的虚构信息（这是最容易犯的错误）。
2. 如果需求缺少关键信息（如连社区名称都没给），先用 2-3 句话列出"需要补充的信息清单"，
   再基于已有信息给出一版带占位符的文案草稿，方便用户填充。
3. 不要捏造数据、奖项、资质、"首家/唯一"等宣传用语。

输出要求：
1. 先给一版正式文案，再给一版朋友圈/社群简短文案。
2. 突出房源亮点与行动号召（联系电话/扫码等用占位符【电话】表示）。
3. 总字数控制在 300 字以内。"""


def generate(question: str) -> dict:
    llm = model_router.chat_model_for("copywriting", temperature=0.7)
    text = llm.invoke(COPY_PROMPT.format(question=question)).content
    return {"intent": "aigc", "answer": text}
