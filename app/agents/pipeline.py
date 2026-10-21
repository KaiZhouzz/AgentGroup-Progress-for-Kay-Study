"""语义澄清 + 错误裁判的数据分析流水线。

四阶段状态机：
  PLAN     语义模型(14B)理解需求，口径不明确则生成带选项的澄清问题反问用户
  FINALIZE 用户回答后，语义模型(14B)把答案合并成确认版 spec
  EXECUTE  code 模型(14B)按 spec 编码 → 沙箱执行；失败由语义模型(14B)裁判
           （mechanical 自动修 / business 歧义升级问用户），循环直到通过
  ADVISE   语义模型(7B)基于执行结果写经营建议

所有 LLM 结构化输出都带解析失败降级，流水线永不卡死。
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field

from app.agents import analysis
from app.core import model_router
from app.parsers import parse_dataframe_summary
from app.sandbox.executor import get_executor

logger = logging.getLogger(__name__)

MAX_ATTEMPTS_PER_ROUND = 4  # 每次 resume 最多自动修复轮数
FINGERPRINT_REPEAT_LIMIT = 2  # 同一错误指纹重复次数达到即升级问用户

PLAN_PROMPT = """你是长租公寓数据分析需求的语义理解专家。用户上传了数据文件并提出分析需求。
请判断需求是否已经足够明确可以直接执行，口径不明确则向用户反问。

需要自查的口径包括：
1. 跨表关联键（如果涉及多份数据拼接：用哪两列关联、格式怎么统一）
2. 指标口径（如坪效 = 收入 ÷ 面积，面积用建筑面积还是使用面积）
3. 时间范围（全部数据还是某几个月）
4. 排除项（如已退租、空置房是否计入）
5. 预测类需求（含"预测/预估/未来/趋势"）：预测哪个指标、预测未来几期、
   用什么方法（趋势外推/移动平均/回归，不确定时作为问题问用户，并在 spec 里写明选定的预测期数）

【用户需求】
{question}

【数据文件清单】（共 {n_files} 个，用文件名键控访问 DATA_FILES["文件名"]，禁止臆造更多文件）
{file_list}

【数据概要】
{schema}

【结构探查结果】（真实执行得出，列名以此为准）
{explore}

【可用列名清单】
{columns}

只输出 JSON，格式如下（不要输出任何其他文字）：
如果需求已明确：
{{"ready": true, "spec": "把需求改写成口径写死的确认版执行指令（写明关联键、指标公式、时间范围、排除项；预测类需求还要写明预测期数和方法）", "questions": []}}
如果不明确：
{{"ready": false, "spec": "", "questions": [
  {{"id": "q1", "question": "具体问题描述", "options": ["选项A", "选项B", "其他"]}}
]}}

要求：
- 最多问 3 个问题，每个问题必须给 2-4 个具体选项（最后一个是"其他"）。
- 问题要具体可答，引用真实列名；不要问数据里已有的信息。
- 如果需求中指明了具体文件名，spec 必须声明【数据源】只使用该文件，并检查该文件
  是否包含分析所需的关键列；缺列时把"缺什么列/该用哪份数据"作为澄清问题问用户，
  禁止在 spec 里臆造与其他文件的关联。
- 如果需求本身已经足够明确（单表统计、口径常识），直接 ready=true，不要为问而问。"""

FINALIZE_PROMPT = """你是长租公寓数据分析需求的语义理解专家。用户之前对澄清问题的回答如下，请把回答合并进需求，输出最终确认版执行指令（spec）。

【原始需求】
{question}

【数据文件清单】（共 {n_files} 个，文件名键控访问，spec 里禁止引用不存在的文件）
{file_list}

【初步 spec（如有）】
{spec}

【澄清问题与用户回答】
{qa_pairs}

【数据概要】
{schema}

【结构探查结果】（列名以此为准）
{explore}

只输出 JSON：
{{"spec": "最终确认版执行指令：写明数据来源、跨表关联键及格式统一方式、指标公式、时间范围、排除项；预测类需求还要写明预测哪个指标、未来几期、用什么方法。措辞要具体到 code 模型不需要再猜。"}}"""

DIAGNOSE_PROMPT = """你是数据分析脚本报错的裁判。判断报错属于哪一类，并给出处理建议。

【执行需求 spec】
{spec}

【失败代码（尾部）】
{code_tail}

【执行报错（已降噪）】
{error}

【可用列名清单】
{columns}

只输出 JSON（不要输出任何其他文字）：
{{"error_class": "mechanical 或 technical 或 business 或 unknown",
  "root_cause": "一句话根因",
  "fix_hint": "给 code 模型的具体修复指令（mechanical/technical 时必填，要具体到改哪一行怎么改）",
  "need_human": false,
  "questions": []}}

判定标准：
- mechanical：代码本身写错了，不依赖业务口径就能修。如 KeyError 列名不存在、
  正则语法错、类型错（str.extract 返回 DataFrame 当 Series 用）、merge 键格式不一致、
  引用了未定义的变量。need_human=false，questions 留空。
- technical：纯技术环境问题，业务用户绝无可能知道答案，必须自动修、禁止问用户。
  如读取引擎不对（xlrd/openpyxl 选择）、用 read_csv 读 xlsx、缺少依赖库、
  文件路径/编码问题。need_human=false，fix_hint 给出确定性修法
  （如改用 pd.read_excel(DATA_FILES["文件名"], engine="openpyxl")）。
  报错含"禁止 import / 禁止 from ... import"时是沙箱白名单拦截（不是缺库）：
  fix_hint 必须指导改用白名单内的库重写（pandas/numpy/matplotlib/sklearn/scipy），
  绝对禁止建议 pip 安装或"解除禁用"——那类修法不存在。
- business：报错暴露的是需求口径歧义，修代码前必须问用户。如：merge 后匹配率为 0
  （关联键构造方式有多种业务解释）、结果为空（过滤条件口径有多种解释）、
  同一指标有多个合理口径。need_human=true，questions 给出 1-2 个带 options 的问题。
  注意：问题必须是业务口径（关联方式/指标公式/范围/排除项），禁止问技术细节。
- unknown：看不出根因。fix_hint 给通用排查建议。

【示例1】KeyError: '月份' → {{"error_class": "mechanical", "root_cause": "列名写错", "fix_hint": "列'月份'不存在，请改用真实列名（见可用列名清单），先 print(df.columns.tolist()) 核对", "need_human": false, "questions": []}}
【示例2】merge 后右表列全为 NaN，匹配率 0 → {{"error_class": "business", "root_cause": "关联键构造方式不确定，匹配率为 0", "fix_hint": "", "need_human": true, "questions": [{{"id": "join_key", "question": "两份数据如何关联？确权表没有房间号列，只有'房源'文本。", "options": ["从'房源'文本提取末尾数字作房号，与房间列表'房间名称'关联", "其他"]}}]}}"""

ESCALATE_PROMPT = """数据分析脚本反复执行失败，自动修复没有奏效。请你向用户提一个"定向澄清问题"，让用户补充能解除卡点的业务口径。

【执行需求】
{spec}

【反复出现的报错】
{error}

【失败代码（尾部）】
{code_tail}

【可用列名清单】
{columns}

只输出 JSON（不要输出任何其他文字）。先判断卡点性质：
- 如果卡点是技术性问题（读取引擎、文件格式、依赖库、路径编码等），用户是业务人员、
  绝无可能知道答案，输出：{{"technical": true}}
- 如果卡点是业务口径问题（关联方式/指标公式/时间范围/排除项），输出：
  {{"technical": false,
    "question": "向用户提的问题。必须包含三部分：1) 用一句话白话说明现在卡在什么；2) 明确指出需要用户补充哪种业务口径（跨表关联键怎么构造/指标公式/时间范围/排除项，按实际卡点选）；3) 提示用户可以选'其他'自填具体口径。",
    "options": ["给出2-3个最可能的口径选项", "其他"]}}

要求：问题必须面向业务口径、具体到列名级别，用户看完就知道该填什么；
禁止问任何技术细节（引擎、库、格式、代码写法）；不要泛泛地问'你想怎么办'。"""


@dataclass
class AnalysisSession:
    session_id: str
    question: str
    data_files: list[str]
    phase: str = "plan"            # plan | execute | done
    spec: str = ""
    schema: str = ""
    explore: str = ""
    columns: str = ""
    questions: list = field(default_factory=list)
    attempts: int = 0
    error_fingerprints: list = field(default_factory=list)
    last_code: str = ""
    last_error: str = ""
    allowed_files: list | None = None   # 需求中指名的文件白名单；None=全部可用
    pending_extra_files: list | None = None  # 白名单误伤待用户放行的额外文件；None=无
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "question": self.question,
            "data_files": self.data_files,
            "phase": self.phase,
            "spec": self.spec,
            "schema": self.schema,
            "explore": self.explore,
            "columns": self.columns,
            "questions": self.questions,
            "attempts": self.attempts,
            "error_fingerprints": self.error_fingerprints,
            "last_code": self.last_code,
            "last_error": self.last_error,
            "allowed_files": self.allowed_files,
            "pending_extra_files": self.pending_extra_files,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisSession":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _parse_json(raw: str) -> dict | None:
    """从 LLM 输出里提取并解析 JSON，失败返回 None（绝不抛异常）。"""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except (ValueError, json.JSONDecodeError):
        return None


def _normalize_questions(raw_questions) -> list[dict]:
    """裁判/PLAN 给出的 questions 可能是非 dict 元素或缺字段，收敛成可用结构。"""
    out = []
    for i, q in enumerate(raw_questions or []):
        if not isinstance(q, dict):
            continue
        question = _as_text(q.get("question"))
        if not question:
            continue
        options = [_as_text(o) for o in (q.get("options") or [])]
        out.append({
            "id": _as_text(q.get("id")) or f"q{i+1}",
            "question": question,
            "options": [o for o in options if o],
        })
    return out


def _as_text(value) -> str | None:
    """7B 模型偶尔把 JSON 的字符串值写成嵌套 dict/list，统一收敛成文本。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _error_fingerprint(error_text: str) -> str:
    """报错指纹：异常类型 + 异常消息 + 用户脚本报错行。

    只用"类型+行号"会误判：不同 KeyError 都穿过 pandas 内部同一帧
    （如 indexes/base.py get_loc），行号相同但根因完全不同。
    """
    exc = re.findall(r"(\w+(?:Error|Exception))\b", error_text)
    exc_type = exc[-1] if exc else "Unknown"
    msg_m = re.findall(r"\w+(?:Error|Exception):\s*(.+)", error_text)
    msg = (msg_m[-1] if msg_m else "?").strip()[:60]
    user_line = re.findall(r'analysis\.py", line (\d+)', error_text)
    line = user_line[-1] if user_line else "?"
    return f"{exc_type}|{msg}|{line}"


def _mentioned_files(question: str, data_files: list[str]) -> list[str] | None:
    """从需求中提取明确指名的数据文件名；命中则只放行这些文件（None=不限制）。

    用户说"分析某某表.xlsx"时，其余已登记文件与本需求无关，
    必须在执行层隔离，防止模型为了凑列引入无关数据。
    """
    from pathlib import Path

    hits = []
    for f in data_files:
        name = Path(f).name
        if name in question:
            hits.append(f)
    return hits or None


def _clean_stderr(stderr: str) -> str:
    """去掉 PerformanceWarning / sandbox 路径等噪音，给报错反馈留有效信息预算。"""
    return "\n".join(
        ln for ln in stderr.splitlines()
        if "Warning" not in ln and "sandbox_" not in ln
    )


def _files_guard_error(code: str, run_files: list[str]) -> str | None:
    """代码引用白名单外的文件名或越界 FILES 下标时给出确定性修正提示（不烧裁判轮次）。"""
    from pathlib import Path

    allowed_names = [Path(f).name for f in run_files]
    bad_keys = [k for k in re.findall(r'DATA_FILES\["([^"]+)"\]', code) if k not in allowed_names]
    if bad_keys:
        return (
            f"KeyError: DATA_FILES 里没有这些键：{bad_keys}。\n"
            f"本次允许读取的文件（DATA_FILES 键名逐字一致）：{allowed_names}。\n"
            "只允许用 DATA_FILES[\"文件名\"] 读取上面列出的文件，禁止臆造文件名、禁止用其他表的 sheet 名。"
        )
    idxs = [int(m.group(1)) for m in re.finditer(r"FILES\[(\d+)\]", code)]
    if idxs and max(idxs) >= len(run_files):
        return (
            f"IndexError: FILES 下标越界。本次一共只有 {len(run_files)} 个数据文件，"
            f"合法下标只有 0..{len(run_files) - 1}，代码引用了 FILES[{max(idxs)}]。\n"
            "确权表的多个月份是同一个文件里的不同 sheet，load_all 已合并全部 sheet，"
            "禁止按月份拆成多个 FILES 下标。"
        )
    return None


def _extra_files_referenced(code: str, run_files: list[str], all_files: list[str]) -> list[str]:
    """代码引用了白名单外、但本次会话确实存在的文件：这些是"需求没点名但分析需要"的候选。"""
    from pathlib import Path

    run_names = {Path(f).name for f in run_files}
    all_by_name = {Path(f).name: f for f in all_files}
    out = []
    for key in re.findall(r'DATA_FILES\["([^"]+)"\]', code):
        if key not in run_names and key in all_by_name and all_by_name[key] not in out:
            out.append(all_by_name[key])
    return out


def build_context(data_files: list[str], timeout: int = 120) -> dict:
    """数据上下文：概要 + 沙箱探查 + 可用列名清单。"""
    schema_blocks = []
    for f in data_files:
        try:
            schema_blocks.append(parse_dataframe_summary(f))
        except Exception as exc:
            schema_blocks.append(f"文件 {f} 概要读取失败: {exc}")
    schema = "\n\n".join(schema_blocks) if schema_blocks else "（无数据文件）"

    explore_res = get_executor().execute(analysis.EXPLORE_SCRIPT, files=data_files, timeout=timeout)
    explore = explore_res.stdout[-3000:] if explore_res.success and explore_res.stdout.strip() else "（探查失败，仅依据数据概要）"
    cols = analysis._available_columns(explore) + analysis._available_columns(schema)
    columns = "、".join(cols) or "（未能解析出列名）"
    return {"schema": schema, "explore": explore, "columns": columns}


class AnalysisPipeline:
    """四阶段状态机。start() / resume(answers) 均可返回 clarify 或最终结果。"""

    def __init__(self, data_files: list[str], timeout: int = 120):
        self.data_files = data_files
        self.timeout = timeout

    # ---------- 对外入口 ----------

    def start(self, question: str) -> dict:
        ctx = build_context(self.data_files, timeout=self.timeout)
        sess = AnalysisSession(
            session_id=uuid.uuid4().hex[:12],
            question=question,
            data_files=self.data_files,
            schema=ctx["schema"],
            explore=ctx["explore"],
            columns=ctx["columns"],
            allowed_files=_mentioned_files(question, self.data_files),
        )
        plan = self._plan(sess)
        if plan is not None and not plan.get("ready") and plan.get("questions"):
            sess.questions = _normalize_questions(plan["questions"])
            if sess.questions:
                return self._clarify_result(sess, "在动手分析之前，有几个口径需要和你确认：")
        sess.spec = _as_text((plan or {}).get("spec")) or question
        sess.phase = "execute"
        return self._execute(sess)

    def resume(self, sess: AnalysisSession, answers: dict[str, str]) -> dict:
        sess.attempts = 0  # 新一轮：重新给满自动修复预算
        sess.error_fingerprints = []
        if sess.pending_extra_files:
            # 上一轮是"白名单外文件放行确认"，不是口径澄清：直接调整文件范围继续执行
            choice = (answers.get("extra_files") or "")
            if "允许" in choice or "全部" in choice:
                logger.info("用户放行额外文件 %s，白名单解除", sess.pending_extra_files)
                sess.allowed_files = None
            else:
                logger.info("用户不允许读取额外文件 %s，保持白名单", sess.pending_extra_files)
            sess.pending_extra_files = None
            sess.questions = []
            sess.last_error = ""
            sess.phase = "execute"
            return self._execute(sess)
        if sess.questions:
            qa = "\n".join(
                f"问：{q.get('question', '')}\n答：{answers.get(q.get('id', ''), '（未回答）')}"
                for q in sess.questions
            )
            merged = self._finalize_spec(sess, qa)
            sess.spec = merged or sess.spec
        sess.questions = []
        sess.last_error = ""  # spec 已更新，旧报错不再适用
        sess.phase = "execute"
        return self._execute(sess)

    # ---------- 阶段实现 ----------

    def _file_list(self) -> str:
        from pathlib import Path

        return "\n".join(f'DATA_FILES["{Path(f).name}"]' for f in self.data_files)

    def _plan(self, sess: AnalysisSession) -> dict | None:
        """PLAN：语义模型判断需求是否明确。JSON 失败返回 None（降级为直接执行）。"""
        llm = model_router.local_planner_model(temperature=0.2)
        prompt = PLAN_PROMPT.format(
            question=sess.question, schema=sess.schema[-2500:],
            explore=sess.explore, columns=sess.columns,
            n_files=len(self.data_files), file_list=self._file_list(),
        )
        try:
            raw = llm.invoke(prompt).content
        except Exception as exc:
            logger.warning("PLAN 调用失败，降级直接执行: %s", exc)
            return None
        plan = _parse_json(raw)
        if plan is None:
            logger.warning("PLAN JSON 解析失败，降级直接执行。原始输出: %.300s", raw)
        return plan

    def _finalize_spec(self, sess: AnalysisSession, qa_pairs: str) -> str | None:
        """FINALIZE：语义模型把用户回答合并进 spec。失败返回 None（保留旧 spec）。"""
        llm = model_router.local_planner_model(temperature=0.2)
        prompt = FINALIZE_PROMPT.format(
            question=sess.question, spec=sess.spec, qa_pairs=qa_pairs,
            schema=sess.schema[-2500:], explore=sess.explore,
            n_files=len(self.data_files), file_list=self._file_list(),
        )
        try:
            raw = llm.invoke(prompt).content
        except Exception as exc:
            logger.warning("FINALIZE 调用失败: %s", exc)
            return None
        merged = _parse_json(raw)
        if merged is None:
            logger.warning("FINALIZE JSON 解析失败，spec 追加问答原文。原始输出: %.300s", raw)
            return sess.spec + "\n【用户确认的口径】\n" + qa_pairs
        return _as_text(merged.get("spec")) or None

    def _diagnose(self, sess: AnalysisSession, error: str) -> dict | None:
        """裁判：语义模型给报错分类。JSON 失败返回 None（走确定性提示降级）。"""
        llm = model_router.local_planner_model(temperature=0.1)
        prompt = DIAGNOSE_PROMPT.format(
            spec=sess.spec[-2000:], code_tail=sess.last_code[-2000:],
            error=error[:1500], columns=sess.columns,
        )
        try:
            raw = llm.invoke(prompt).content
        except Exception as exc:
            logger.warning("diagnose 调用失败，走确定性提示: %s", exc)
            return None
        diag = _parse_json(raw)
        if diag is None:
            logger.warning("diagnose JSON 解析失败，走确定性提示。原始输出: %.300s", raw)
        return diag

    def _escalate_question(self, sess: AnalysisSession) -> list[dict] | None:
        """升级提问：优先语义模型生成定向问题（带报错上下文）。

        返回 None 表示卡点是技术性问题，不该打扰业务用户。
        """
        llm = model_router.local_planner_model(temperature=0.2)
        prompt = ESCALATE_PROMPT.format(
            spec=sess.spec[-1500:], error=sess.last_error[:1200],
            code_tail=sess.last_code[-1200:], columns=sess.columns,
        )
        try:
            raw = llm.invoke(prompt).content
            parsed = _parse_json(raw)
            if parsed and parsed.get("technical"):
                logger.info("卡点为技术性问题，不问用户，走失败收尾")
                return None
            if parsed and parsed.get("question"):
                q = {
                    "id": "escalate",
                    "question": _as_text(parsed["question"]),
                    "options": [_as_text(o) for o in (parsed.get("options") or []) if _as_text(o)],
                }
                return [q]
        except Exception as exc:
            logger.warning("escalate 提问生成失败: %s", exc)
        return None

    def _needs_chart(self, sess: AnalysisSession) -> bool:
        """需求/口径里是否包含绘图要求（饼图/柱状/折线/可视化等）。"""
        text = sess.question + "\n" + sess.spec
        specific = ("饼图", "柱状", "条形", "折线", "散点", "曲线", "可视化", "直方")
        return any(k in text for k in specific) or ("画" in text and "图" in text)

    def _execute(self, sess: AnalysisSession) -> dict:
        """EXECUTE：code 模型编码 → 沙箱执行 → 失败裁判修复，循环直到成功或升级。"""
        # 需求指名了文件 → 执行层硬隔离：沙箱只见这些文件，模型想拉无关表会被 guard 打回
        from pathlib import Path

        run_files = sess.allowed_files or self.data_files
        allowed_desc = "\n".join(
            f'DATA_FILES["{f.split("/")[-1]}"]' for f in run_files
        )
        chart_retried = False
        for _ in range(MAX_ATTEMPTS_PER_ROUND):
            temperature = 0.1 if sess.attempts == 0 else 0.3
            llm = model_router.chat_model_for("code", temperature=temperature)
            prompt = analysis.CODE_GEN_PROMPT.format(
                schema=sess.schema, explore=sess.explore, columns=sess.columns,
                question=sess.spec, n_files=len(run_files), allowed=allowed_desc,
            )
            if sess.last_error:
                prompt += (
                    "\n\n【上一次生成的脚本执行报错，请修正后重新输出完整脚本。"
                    "检查是否使用了不存在的列名或表结构理解错误。】\n"
                    f"报错信息：\n{sess.last_error[:1500]}"
                    + analysis._column_hint(sess.last_error, sess.explore)
                    + "\n注意：仍必须以上一轮同样的方式开头——报表型文件一律用 "
                      "df = load_all(DATA_FILES[\"文件名\"])，禁止改回 pd.read_excel 直接读。"
                )
            raw = llm.invoke(prompt).content
            code = analysis._fix_imports(analysis._extract_code(raw))
            code = analysis._repair_column_names(code, sess.explore + "\n" + sess.schema)
            code = analysis._repair_read_calls(code, run_files)
            code = analysis._repair_sheet_names(code, run_files)
            sess.last_code = code
            sess.attempts += 1
            logger.info("generated code (attempt %d):\n%s", sess.attempts, code)

            # 白名单外文件名/越界下标是确定性错误，不必跑沙箱也不烧裁判轮次
            guard_err = _files_guard_error(code, run_files)
            if guard_err:
                sess.last_error = guard_err
                logger.warning("files guard triggered (attempt %d)", sess.attempts)
                # 需求只点名了部分文件，但分析确实需要其他文件里的数据（如面积在房间表里）：
                # 不问清楚就永远卡在这里，直接问用户是否放行（业务问题，不是技术问题）
                extra = _extra_files_referenced(code, run_files, self.data_files)
                if extra:
                    mentioned = [Path(f).name for f in run_files]
                    sess.pending_extra_files = extra
                    sess.questions = [{
                        "id": "extra_files",
                        "question": (
                            f"分析还需要用到 {'、'.join(Path(f).name for f in extra)} 里的数据"
                            f"（你只提到了 {'、'.join(mentioned)}）。要把这些文件也纳入本次分析吗？"
                        ),
                        "options": ["允许，全部纳入分析", "不允许，只用我提到的文件"],
                    }]
                    return self._clarify_result(sess, "分析需要用到需求之外的数据文件，需要你确认：")
                continue

            result = get_executor().execute(code, files=run_files, timeout=self.timeout)
            if result.success:
                # 需求要求画图但脚本没出图（模型修复报错时常把绘图部分"减负"丢掉）→ 补一轮
                if self._needs_chart(sess) and not result.images and not chart_retried:
                    chart_retried = True
                    sess.last_error = (
                        "脚本执行成功但没有生成图表。原需求包含绘图要求，请在完整脚本里保留/补充："
                        "matplotlib 使用 Agg 后端，绘图后 plt.savefig(OUT_DIR + '/chart.png')。"
                        "数据处理逻辑保持不变，只加绘图部分。"
                    )
                    logger.info("需求含绘图但脚本未出图，追加一轮补图")
                    continue
                return self._advise(sess, result)

            clean_err = _clean_stderr(result.stderr)
            sess.last_error = ((result.error or "") + "\n" + clean_err).strip() or "未知错误"
            logger.warning("execution failed (attempt %d): %s", sess.attempts, sess.last_error[:800])

            # 裁判：分类 + 修复提示
            diag = self._diagnose(sess, sess.last_error)
            fix_hint = ""
            if diag:
                cls = diag.get("error_class", "unknown")
                if diag.get("need_human") or cls == "business":
                    questions = _normalize_questions(diag.get("questions"))
                    if not questions:
                        questions = self._escalate_question(sess) or self._escalate_fallback(sess)
                    sess.questions = questions
                    return self._clarify_result(
                        sess,
                        f"执行卡住了，需要你做个决策：\n（报错：{sess.last_error.splitlines()[-1][:200]}）",
                    )
                fix_hint = _as_text(diag.get("fix_hint")) or ""
                logger.info("diagnose: %s, fix_hint=%.300s", cls, fix_hint)
            else:
                fix_hint = analysis._column_hint(sess.last_error, sess.explore)

            # 同一错误指纹重复出现 → 机械修复无效，生成定向问题问用户
            fp = _error_fingerprint(sess.last_error)
            sess.error_fingerprints.append(fp)
            if sess.error_fingerprints.count(fp) >= FINGERPRINT_REPEAT_LIMIT:
                questions = self._escalate_question(sess)
                if questions is None:
                    return self._fail_result(sess)  # 技术性卡点，不打扰业务用户
                sess.questions = questions
                return self._clarify_result(sess, "同样的错误反复出现，需要你补充一个业务口径才能继续：")

            if fix_hint:
                sess.last_error += "\n\n【裁判修复提示】\n" + fix_hint[:800]
            sess.updated_at = time.time()

        return self._fail_result(sess)

    def _fail_result(self, sess: AnalysisSession) -> dict:
        """执行失败的收尾：技术性卡点也走这里，面向用户给出可操作建议。"""
        err_tail = sess.last_error[-600:]
        return {
            "intent": "analysis",
            "answer": (
                f"分析脚本执行失败（已自动修复 {MAX_ATTEMPTS_PER_ROUND} 轮仍未通过，报错属于技术性问题，已自动处理但未奏效）。\n\n"
                f"报错摘要：\n```\n{err_tail}\n```\n\n"
                "建议：换个说法重新描述需求（如指出用哪张表的哪一列做关联），或补充数据后重试。"
            ),
            "code": sess.last_code,
            "stdout": "",
            "stderr": sess.last_error[:1000],
            "success": False,
            "images": [],
        }

    def _escalate_fallback(self, sess: AnalysisSession) -> list[dict]:
        """escalate 生成失败时的兜底问题（面向业务口径）。"""
        err_line = sess.last_error.splitlines()[-1][:150]
        return [{
            "id": "escalate",
            "question": (
                f"脚本反复卡在同一个报错：{err_line}。\n"
                "这通常意味着缺一个业务口径才能继续。请在'其他'里补充说明（越具体越好）：\n"
                "· 如果涉及多份数据拼接：用什么列关联？格式怎么统一？（如：确权表'房源'末尾数字 = 房间列表'房间名称'）\n"
                "· 指标公式：分子分母分别是哪列？（如：坪效 = 收入 ÷ 建筑面积(㎡)）\n"
                "· 时间范围/排除项：算哪段数据？要不要排除退租房源？"
            ),
            "options": ["按数据概要里的列名补充口径（选其他并填写）", "放弃本次分析"],
        }]

    def _advise(self, sess: AnalysisSession, result) -> dict:
        """ADVISE：基于执行结果写经营建议。"""
        sess.phase = "done"
        if result.stdout.strip():
            advisor = model_router.chat_model_for("chat", temperature=0.4)
            advice = advisor.invoke(analysis.ADVICE_PROMPT.format(
                question=sess.question,
                stdout=result.stdout[-4000:],
                stderr=result.stderr[-1500:],
            )).content
        else:
            advice = "脚本执行成功但没有输出。请补充更明确的分析指标或检查数据内容。"
        return {
            "intent": "analysis",
            "answer": advice,
            "code": sess.last_code,
            "stdout": result.stdout,
            "stderr": "",
            "success": True,
            "images": result.images,
        }

    def _clarify_result(self, sess: AnalysisSession, headline: str) -> dict:
        sess.updated_at = time.time()
        return {
            "intent": "clarify_analysis",
            "answer": headline,
            "questions": sess.questions,
            "session": sess.to_dict(),
            "success": True,
            "images": [], "code": "", "stdout": "", "stderr": "",
        }
