"""数据分析 agent：LLM 生成 pandas 脚本 → 沙箱执行 → 基于结果写经营建议。"""
from __future__ import annotations

import logging
import re

from app.core import model_router
from app.parsers import parse_dataframe_summary
from app.sandbox.executor import ExecutionResult, get_executor

logger = logging.getLogger(__name__)

CODE_GEN_PROMPT = r"""你是长租公寓经营数据分析师。根据用户需求和数据概要，生成一段完整可执行的 Python 脚本。

【硬性要求】
1. 脚本第一行从 data_files 模块读取文件路径（模板已给出，不要改）：
   一律用文件名键控访问 DATA_FILES["文件名"]，文件名逐字照抄【数据源约束】里列出的名字，
   绝对禁止用 FILES[i] 下标——下标顺序不可预测，用下标必然读错文件。
2. 只读数据文件，不要修改/删除任何文件；不要联网；不要使用 os/sys/subprocess/socket 等模块。
3. 只能用 pandas / numpy / matplotlib / openpyxl / json / math / statistics / datetime / csv / re / io / collections / itertools / functools。
4. 结果用 print() 输出关键数字，文字结论也用 print 输出。
5. 如需画图：matplotlib 使用 Agg 后端，图片保存到 OUT_DIR 下的 png 文件，并设置中文字体兜底：
   import matplotlib
   matplotlib.use("Agg")
   import matplotlib.pyplot as plt
   try:
       plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
       plt.rcParams['axes.unicode_minus'] = False
   except Exception:
       pass
6. 列名一律以【数据概要】和【结构探查结果】中出现的为准，只允许使用那里出现过的列名，
   禁止使用臆测的列名；读取数据后第一行先 print(df.columns.tolist()) 自检。
7. 若【结构探查结果】中出现【推荐开头代码】，说明这是多层表头报表：你的脚本必须原样以
   该代码块开头（后面可 import 其他允许的库），用 df = load_all(DATA_FILES["文件名"]) 得到
   合并全部 sheet 且已修正表头的数据框，不得再自行 pd.read_excel；列名以探查输出的"修正后列名"为准。
8. 同名列已按探查代码自动加 _2、_3 后缀区分，直接用带后缀的列名。
9. 修正表头后的列默认都是字符串类型；若某列转数值失败，load_all 已自动处理，
   直接对其 sum()/mean() 即可；不要对整列做字符串拼接式统计。
10. pandas 陷阱：s.str.extract(正则) 默认返回 DataFrame；单分组取值要用
    s.str.extract(正则, expand=False)（或取 [0]）。groupby.agg 里自己命名的新列
    （如 应收=("实际租金","sum")）后续用 monthly["应收"] 引用，名字必须前后一致；
    写完后请核对引用的列名与定义完全一致。
11. 从混合文本取数字必须容错（文本不规整，如"…-2单元-0410室"末尾带"室"）：
    用 extract 取不到时返回 NaN 而不是报错，例如
    df["房号"] = df["房源"].str.extract(r"(\d+)(?!.*\d)", expand=False)
    （取最后一个数字串；失败为 NaN，后续可用 .fillna("0") 处理），
    禁止用 int(x.split(...)) 这类遇到脏数据直接崩溃的写法。
12. 只有数值列才能 sum()/mean()：字符串列会被 groupby().sum() 静默丢弃导致后续 KeyError。
    若 load_all 没能把某列转成数值（如带特殊格式），显式转换后再算：
    df["列名"] = pd.to_numeric(df["列名"], errors="coerce")
13. 任何列在用 .str 之前必须先确认是字符串类型：非字符串列（如房间名称可能是
    int64）直接 .str 会报 AttributeError。保险写法统一加一步：
    df["列名"] = df["列名"].astype(str)
14. 用 + 拼接多列构造关联键时，必须每一列都转成字符串并填掉空值，
    只要有一列是 int64/NaN 就会触发 pyarrow 崩溃
    （ArrowNotImplementedError: binary_join_element_wise）。标准写法：
    df["键"] = (df["列A"].astype("string").fillna("") + "-" +
                df["列B"].astype("string").fillna("").str.lstrip("0"))
    另外构造的列名要全程一致：赋值叫什么，merge 的 on 就必须用什么名，
    禁止赋值用 "单元&房号"、合并却写 "单元&房间名称"。
15. 【数据源约束】本次分析只允许读取以下文件，禁止读取未列出的任何文件：
    {allowed}
    sheet 名只允许用【结构探查结果】里 "sheets:" 打印过的真实 sheet 名，禁止臆造 sheet 名；
    如果需要的列在允许的文件里不存在，print 说明缺了哪一列后正常结束
    （系统会向用户追问），绝对禁止引入其他文件"凑列"。

【脚本开头】二选一：
A. 若【结构探查结果】有【推荐开头代码】：原样复制该代码块作为脚本开头（见规则 7）。
B. 否则用此模板开头：
from data_files import DATA_FILES, OUT_DIR
import pandas as pd

df = pd.read_excel(DATA_FILES["文件名.xlsx"])   # 键名必须逐字照抄【数据源约束】里的文件名

【多文件说明】
- 本次共 {n_files} 个数据文件，DATA_FILES 可用键：
  {allowed}
- 每个文件用 DATA_FILES["文件名"] 精确取路径，禁止 FILES 下标、禁止猜文件名；
  禁止引用【数据源约束】之外的任何文件。
- 若 {n_files} 为 1：禁止对 DATA_FILES 做任何循环，直接 df = load_all(DATA_FILES["文件名"]) 分析即可。
- 若涉及多文件拼接：分别按文件名读取，再用 pd.merge / pd.concat 关联，例如：
  df1 = pd.read_excel(DATA_FILES["文件A.xlsx"]); df2 = pd.read_excel(DATA_FILES["文件B.xlsx"])
  merged = pd.merge(df1, df2, on="房间号", how="left")
- 关联键（如房间号、合同号）优先用两份数据中都存在的列；列名一律以数据概要为准，不要臆测。
- 跨表合并前必须先统一关联键格式（类型/前导零/空格都会导致合并失败或匹配为 0）：
  for d, k in [(df1, "键列1"), (df2, "键列2")]:
      d[k] = d[k].astype(str).str.strip().str.lstrip("0")
- 若一边是全称（如"xx社区-2单元-0447"）另一边是短号（如 1001），不能直接合并：
  要么截取/拼接出相同格式的键（如 单元 + 房号），要么用 str.extract(r"(\d+)$") 取末尾数字
  再统一格式后合并；合并后必须 print(len(df1), merged["右表列"].notna().sum())
  校验匹配率，匹配为 0 说明键不对，要换构造方式。

【结构探查结果】（真实执行得出，列名/表头以此为准）
{explore}

【可用列名清单】（只能原样使用下面这些列名，一个标点都不能改）
{columns}

【数据概要】
{schema}

【指标口径参考】
- 出租率 = 已出租房间数 / 总房间数
- 欠费率 = 欠费金额 / 应收金额
- 收缴率 = 实收金额 / 应收金额

【预测类需求】（需求含"预测/预估/未来/趋势外推"时适用，必须真建模、出具体数字）
16. 禁止输出"此处仅展示计算结果"之类的占位文字——预测必须真的算出来并 print 具体数值。
17. 方法按数据量选择（白名单内）：
    - 月度序列 ≥ 6 期：numpy.polyfit(月份序号, 指标值, 1) 或 (…, 2) 做趋势外推，
      逐期 print 未来 3-6 期的预测值（带月份标签）；
    - 数据点少：用 pandas rolling(3).mean() 移动平均给出下期预测；
    - 多因素回归用 sklearn.linear_model.LinearRegression。
18. 预测前必须先把历史序列按时间排序并 print 出来（让用户能看到趋势依据），
    再 print 预测值；若用了 sklearn 需 from sklearn.linear_model import LinearRegression
    （sklearn 在白名单内，禁止因 import 被拦就放弃预测，换 polyfit/rolling 也要做出来）。

【few-shot 示例】
需求：统计每月收缴率
生成的脚本：
from data_files import DATA_FILES, OUT_DIR
import pandas as pd

df = pd.read_excel(DATA_FILES["账单.xlsx"])
monthly = df.groupby("月份").agg(应收=("应收", "sum"), 实收=("实收", "sum"))
monthly["收缴率"] = (monthly["实收"] / monthly["应收"] * 100).round(2)
print(monthly)
print(f"整体收缴率: {{(df['实收'].sum() / df['应收'].sum() * 100):.2f}}%")

【用户需求】
{question}

只输出脚本本身，不要输出任何解释、markdown 代码块标记或其他文字。"""


# 报表型 Excel 的读取修正代码：修表头、合并全部 sheet、重复列去重、数值列转换。
# 探查时打印给模型照抄；模型漏抄时由 _fix_imports 自动注入，保证 load_all 必定可用。
PROLOGUE_CODE = '''import pandas as pd
def load_all(path):
    xl = pd.ExcelFile(path)
    out = []
    for sn in xl.sheet_names:
        d = xl.parse(sn)
        if sum(str(c).startswith("Unnamed") for c in d.columns) > len(d.columns) / 2:
            d.columns = d.iloc[0]
            d = d.iloc[1:]
        d = d.reset_index(drop=True)
        d["数据来源sheet"] = sn
        out.append(d)
    df_all = pd.concat(out, ignore_index=True)
    df_all.columns = [str(c) if pd.notna(c) else f"未命名列_{i}" for i, c in enumerate(df_all.columns)]
    seen = {}
    new_cols = []
    for c in df_all.columns:
        seen[c] = seen.get(c, 0) + 1
        new_cols.append(c if seen[c] == 1 else str(c) + "_" + str(seen[c]))
    df_all.columns = new_cols
    for c in df_all.columns:
        s = df_all[c].astype("string").str.strip().str.replace(",", "", regex=False)
        denom = s.notna().sum()
        if not denom:
            continue
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().sum() / denom >= 0.9:
            df_all[c] = num
            continue
        m = s.str.extract(r"(-?\\d+(?:\\.\\d+)?)", expand=False)
        num2 = pd.to_numeric(m, errors="coerce")
        joined = s.dropna().str.cat()
        digit_ratio = sum(ch.isdigit() for ch in joined) / max(len(joined), 1)
        if num2.notna().sum() / denom >= 0.95 and digit_ratio >= 0.6:
            df_all[c] = num2
    return df_all
'''

EXPLORE_SCRIPT = r'''
import pandas as pd
from data_files import DATA_FILES, OUT_DIR

PROLOGUE = """''' + PROLOGUE_CODE + '''"""

for name, f in DATA_FILES.items():
    print(f"===== 文件: {name} =====")
    try:
        if f.lower().endswith(".csv"):
            df = pd.read_csv(f)
            print(f"shape: {df.shape}")
            print("列(前40):", [str(c) for c in df.columns[:40]])
            print(df.head(2).to_string(index=False, max_cols=8)[:400])
            continue
        xl = pd.ExcelFile(f)
        print("sheets:", xl.sheet_names)
        first = xl.parse(xl.sheet_names[0])
        unnamed = sum(str(c).startswith("Unnamed") for c in first.columns)
        print(f"首个sheet [{xl.sheet_names[0]}] shape={first.shape} 未命名列 {unnamed}/{len(first.columns)}")
        if unnamed > len(first.columns) / 2:
            print("检测到多层表头报表。【推荐开头代码】（你的分析脚本必须原样以此开头）：")
            print(PROLOGUE)
            fixed = first.copy()
            fixed.columns = fixed.iloc[0]
            fixed = fixed.iloc[1:]
            seen = {}
            cols = []
            for c in fixed.columns:
                seen[c] = seen.get(c, 0) + 1
                cols.append(c if seen[c] == 1 else str(c) + "_" + str(seen[c]))
            fixed.columns = cols
            print("修正后列名(前40):", [str(c) for c in fixed.columns[:40]])
            print(fixed.head(2).to_string(index=False, max_cols=8)[:400])
        else:
            print("列(前40):", [str(c) for c in first.columns[:40]])
            print(first.head(2).to_string(index=False, max_cols=8)[:400])
    except Exception as e:
        print("读取失败:", e)
'''


def _extract_code(text: str) -> str:
    """从 LLM 输出中提取 Python 代码，兼容 ```python ...``` 包裹。"""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    code = m.group(1) if m else text
    # 去掉模型常见的前缀废话行
    lines = [ln for ln in code.splitlines() if not ln.strip().startswith(("这是", "以下是", "好的"))]
    return "\n".join(lines).strip()


def _fix_imports(code: str) -> str:
    """本地小模型常见遗漏的兜底：补 data_files 导入、matplotlib Agg 后端、load_all 定义。"""
    if "load_all" in code and "def load_all" not in code:
        code = PROLOGUE_CODE + "\n" + code
    if "data_files" not in code:
        code = "from data_files import DATA_FILES, FILES, OUT_DIR\n" + code
    # 用到 plt. 却完全没导入 pyplot：直接补全导入（7B 修复轮次里会反复漏写，
    # 等它自己改不如确定性注入）。已有 matplotlib 导入但缺 Agg 后端时只补后端。
    if re.search(r"\bplt\.", code) and "pyplot" not in code:
        code = "import matplotlib\nmatplotlib.use(\"Agg\")\nimport matplotlib.pyplot as plt\n" + code
    elif "matplotlib" in code and "matplotlib.use" not in code:
        code = "import matplotlib\nmatplotlib.use(\"Agg\")\n" + code
    if re.search(r"\bnp\.", code) and "numpy" not in code:
        code = "import numpy as np\n" + code
    return code


ADVICE_PROMPT = """你是长租公寓经营顾问。以下是对用户问题的数据分析执行结果。

【用户需求】
{question}

【数据执行 stdout】
{stdout}

【stderr（若有）】
{stderr}

请基于以上数字结果写经营建议：
1. 先用 2-3 句话总结数据反映的核心情况（引用具体数字）。
2. 给出 3-5 条可落地的经营建议，按优先级排序。
3. 如果执行失败或结果为空，请说明数据可能的问题并建议用户补充什么数据。
用中文回答。"""


def _available_columns(explore: str) -> list[str]:
    """从探查输出里解析出所有出现过的列名（数据概要和探查结果都会打印列清单）。"""
    import ast

    cols: list[str] = []
    for m in re.finditer(r"(?:列\(清洗后\)|列\(前40\)|修正后列名\(前40\)): (\[.*?\])", explore):
        try:
            cols.extend(str(c) for c in ast.literal_eval(m.group(1)))
        except (ValueError, SyntaxError):
            continue
    seen: set[str] = set()
    return [c for c in cols if not (c in seen or seen.add(c))]


def _column_hint(error_text: str, explore: str) -> str:
    """从 KeyError 里找出模型写错的列名，用模糊匹配给出真实可用列，作为重试修正提示。"""
    import difflib

    m = re.search(r"KeyError:\s*[\['\"]+([^'\]\"\n]+)", error_text)
    if m:
        bad = m.group(1).strip()
        candidates = _available_columns(explore)
        if candidates:
            close = difflib.get_close_matches(bad, candidates, n=4, cutoff=0.3)
            related = [c for c in candidates if bad[:2] in c and c not in close][:4]
            sugg = close + [c for c in related if c not in close]
            if sugg:
                return (
                    f"\n提示：列 '{bad}' 不存在。实际数据里最接近的列：{sugg}。"
                    "请改用真实列名；如果你需要的是组合信息（如 单元+房号），请在脚本里先构造该列再使用。"
                )
        return f"\n提示：列 '{bad}' 不存在，请从数据概要的列清单里选择真实列名。"

    # 合并键类型不匹配（如 str vs int64）：必须先统一两边键的格式
    if "trying to merge on" in error_text or ("merge" in error_text and "ValueError" in error_text):
        return (
            "\n提示：合并失败是因为两边关联键类型/格式不一致（如一边是'xx社区-2单元-0447'，"
            "另一边是数字 1001）。正确做法：先分别构造相同格式的字符串键再合并，例如\n"
            "df1['键'] = df1['单元'].astype(str) + '-' + df1['房号'].astype(str).str.lstrip('0')\n"
            "df2['键'] = df2['单元'].astype(str) + '-' + df2['房间名称'].astype(str).str.lstrip('0')\n"
            "merged = pd.merge(df1, df2, on='键', how='left')"
        )

    # pyarrow 字符串拼接崩溃：+ 两边只要有一列不是字符串类型（int64/NaN）就触发
    if "binary_join_element_wise" in error_text or "ArrowNotImplementedError" in error_text:
        return (
            "\n提示：拼接关联键时用了 + 但某一列不是字符串类型（如房间名称是 int64），"
            "pyarrow 后端会直接崩溃。标准写法（每列都转字符串并填掉空值）：\n"
            "df['键'] = (df['列A'].astype('string').fillna('') + '-' +\n"
            "            df['列B'].astype('string').fillna('').str.lstrip('0'))\n"
            "同时检查：构造列的名字赋值和 merge 的 on 必须完全一致。"
        )

    # 沙箱静态检查拦截（禁止 import）：白名单内的库检查导入写法；
    # 白名单外的库禁止再尝试，直接用白名单内工具改写（绝不可能是"pip 装库"能解决的）
    if "禁止 import" in error_text or "禁止 from ... import" in error_text:
        from app.sandbox.executor import _ALLOWED_IMPORT_ROOTS
        allowed = sorted(_ALLOWED_IMPORT_ROOTS - {"data_files"})
        return (
            f"\n提示：沙箱安全白名单拦截了 import。允许使用的库只有：{allowed}。"
            "被拦截的库禁止再尝试导入（装库也无效），请用白名单内的库改写："
            "线性/多项式回归用 numpy.polyfit、numpy.linalg.lstsq；机器学习用 sklearn；"
            "统计聚合用 pandas groupby/pivot_table。"
        )

    # groupby+reset_index 后做分组占比：total_counts[df["键"]] 返回以键值为索引的 Series，
    # 与 df 的默认 RangeIndex 对不齐，除法时触发 Length of values does not match length of index
    if "does not match length of index" in error_text or "Length of values" in error_text:
        return (
            "\n提示：分组占比/归一化时报索引长度不匹配，是因为 total_counts[df[\"键\"]] 返回的 "
            "Series 以键值为索引，与 df 的默认索引对不齐。改成 .map() 按键取值（保留行序和索引）：\n"
            "df[\"占比\"] = df[\"频数\"] / df[\"键\"].map(total_counts)\n"
            "或在一开始就用 transform 避免 reset_index 后的对齐问题：\n"
            "df[\"组内合计\"] = df.groupby(\"键\")[\"列\"].transform(\"sum\")"
        )
    return ""


def _real_sheets(data_files: list[str]) -> dict[str, list[str]]:
    """缓存各数据文件的真实 sheet 名列表（供 sheet 名校验用）。"""
    import pandas as pd
    from pathlib import Path

    out: dict[str, list[str]] = {}
    for f in data_files:
        try:
            out[Path(f).name] = pd.ExcelFile(f).sheet_names
        except Exception:
            pass
    return out


def _repair_read_calls(code: str, data_files: list[str]) -> str:
    """确定性修正文件读取调用：按真实扩展名纠正 read_csv/read_excel 和引擎参数。

    模型常见错误：对 .xlsx 用 pd.read_csv、对 xlsx 指定 engine="xlrd"
    （xlrd 只支持老的 .xls），都会触发"格式无法确定/引擎错误"——
    这类纯技术问题不该问用户，直接机械修正。
    """
    # 数据里有 xlsx 时，代码里的 xlrd 引擎一律换成 openpyxl
    if any(f.lower().endswith((".xlsx", ".xlsm")) for f in data_files):
        code = code.replace('engine="xlrd"', 'engine="openpyxl"')

    excel_names = {f.split("/")[-1] for f in data_files if f.lower().endswith((".xlsx", ".xlsm"))}

    def fix_keyed(m: re.Match) -> str:
        func, name = m.group(1), m.group(2)
        is_excel = name in excel_names
        if func == "read_csv" and is_excel:
            return f'pd.read_excel(DATA_FILES["{name}"], engine="openpyxl"'
        if func == "read_excel" and not is_excel:
            return f'pd.read_csv(DATA_FILES["{name}"]'
        return m.group(0)

    code = re.sub(r'pd\.(read_csv|read_excel)\(DATA_FILES\["([^"]+)"\]', fix_keyed, code)

    # 旧式 FILES[i] 写法兜底（模型偶尔仍会用）
    def fix_indexed(m: re.Match) -> str:
        func, idx = m.group(1), int(m.group(2))
        try:
            fname = data_files[idx]
        except IndexError:
            return m.group(0)
        is_excel = fname.lower().endswith((".xlsx", ".xlsm"))
        if func == "read_csv" and is_excel:
            return f'pd.read_excel(FILES[{idx}], engine="openpyxl"'
        if func == "read_excel" and not is_excel:
            return f"pd.read_csv(FILES[{idx}]"
        return m.group(0)

    return re.sub(r"pd\.(read_csv|read_excel)\(FILES\[(\d+)\]", fix_indexed, code)


def _repair_file_keys(code: str, data_files: list[str]) -> str:
    """确定性修正臆造的文件名键：只允许真实上传过的文件名（basename）。

    本地小模型最高频的死法：把单文件多月度数据（sheet 名是 202504、202505…）
    臆造成"确权模拟数据_202511.xlsx"这样的分月文件，DATA_FILES 字典直接
    KeyError；而多轮纠错反馈只会让它换着花样继续猜文件名。
    这里按真实文件名的前缀/相似度机械替换，不烧重试轮次。
    """
    import difflib
    from pathlib import Path

    real_names = {Path(f).name for f in data_files}
    stems = {Path(f).stem: Path(f).name for f in data_files}

    def fix(m: re.Match) -> str:
        key = m.group(1)
        if key in real_names:
            return m.group(0)
        key_stem = Path(key).stem
        # 真实文件 stem 是臆造 stem 的前缀（如 确权模拟数据 ⊂ 确权模拟数据_202511）：
        # 取最长前缀对应的文件；前缀不唯一时退回完整名相似度
        prefixes = sorted((s for s in stems if key_stem.startswith(s)), key=len, reverse=True)
        if len(prefixes) == 1:
            return f'DATA_FILES["{stems[prefixes[0]]}"]'
        close = difflib.get_close_matches(key, list(real_names), n=1, cutoff=0.6)
        if close:
            return f'DATA_FILES["{close[0]}"]'
        return m.group(0)

    return re.sub(r'DATA_FILES\["([^"]+)"\]', fix, code)


def _repair_sheet_names(code: str, data_files: list[str]) -> str:
    """确定性修正臆造的 sheet 名：只允许真实存在的 sheet；单 sheet 文件直接去掉 sheet_name 参数。

    模型会把其他文件里见过的 sheet 名（如"房间列表-xxxxx"）张冠李戴到当前文件上，
    触发 Worksheet named 'xxx' not found——纯技术错配，机械修正，不烧裁判轮次。
    """
    sheets_map = _real_sheets(data_files)

    def drop_bad_sheet(m: re.Match) -> str:
        head, name, sheet = m.group(1), m.group(2), m.group(3)
        real = sheets_map.get(name)
        if not real or sheet in real:
            return m.group(0)
        if len(real) == 1:
            return head  # 该文件只有一个 sheet，去掉 sheet_name 参数
        return m.group(0)

    code = re.sub(
        r'(pd\.read_excel\(\s*DATA_FILES\["([^"]+)"\]\s*),\s*sheet_name\s*=\s*["\']([^"\']*)["\']',
        drop_bad_sheet, code,
    )
    code = re.sub(
        r'(pd\.read_excel\(\s*FILES\[(\d+)\]\s*),\s*sheet_name\s*=\s*["\']([^"\']*)["\']',
        lambda m: _fix_indexed_sheet(m, data_files, sheets_map), code,
    )
    return code


def _fix_indexed_sheet(m: re.Match, data_files: list[str], sheets_map: dict) -> str:
    head, idx_s, sheet = m.group(1), m.group(2), m.group(3)
    try:
        from pathlib import Path
        name = Path(data_files[int(idx_s)]).name
    except IndexError:
        return m.group(0)
    real = sheets_map.get(name)
    if not real or sheet in real:
        return m.group(0)
    if len(real) == 1:
        return head
    return m.group(0)


def _repair_column_names(code: str, explore: str) -> str:
    """确定性修正代码里的列名错别字：把与真实列名近似但不存在的字符串字面量替换掉。

    模型最高频的错误就是'月度'写成'月份'这类单字差异，且多轮反馈也常视而不见；
    这里用模糊匹配机械修正（限定短字符串，避免误伤 print 文案等长文本）。
    """
    import difflib

    cols = _available_columns(explore)
    if not cols:
        return code

    def fix(m: re.Match) -> str:
        quote = m.group(1) or m.group(3)
        s = m.group(2) or m.group(4)
        if len(s) > 8 or s in cols:
            return m.group(0)
        # 首字相同 + 相似度≥0.5：覆盖"月度/月份"这类单字差异（其 ratio 恰为 0.5），
        # 同时避免"实收"被误改成"应收"这种语义不同的替换
        close = difflib.get_close_matches(s, cols, n=1, cutoff=0.5)
        if close and close[0][0] == s[0]:
            return f"{quote}{close[0]}{quote}"
        return m.group(0)

    return re.sub(r"(['\"])([^'\"]+)\1", fix, code)


def analyze(question: str, data_files: list[str], timeout: int = 120, max_attempts: int = 4) -> dict:
    """完整分析流水线：生成代码 → 沙箱执行 → 写建议。"""
    schema_blocks = []
    for f in data_files:
        try:
            schema_blocks.append(parse_dataframe_summary(f))
        except Exception as exc:
            schema_blocks.append(f"文件 {f} 概要读取失败: {exc}")
    schema = "\n\n".join(schema_blocks) if schema_blocks else "（无数据文件，请提醒用户先上传账单等数据）"

    # 结构探查：先用固定脚本在沙箱里真实读一遍文件，把真实的 sheet/列名/表头喂给模型，
    # 避免报表型 Excel（合并单元格、Unnamed 列）导致模型臆测列名
    explore_res = get_executor().execute(EXPLORE_SCRIPT, files=data_files, timeout=timeout)
    explore = explore_res.stdout[-3000:] if explore_res.success and explore_res.stdout.strip() else "（探查失败，仅依据数据概要）"
    columns = "、".join(_available_columns(explore) + _available_columns(schema)) or "（未能解析出列名）"

    # 代码生成 → 沙箱执行 → 失败把报错喂回模型继续修，多轮迭代直到成功或耗尽次数。
    # 本地小模型单轮成功率有限，多轮自我修复是主要兜底手段。
    # 温度首轮低（求稳），后续略升（避免生成雷同的错误代码）；每轮都锚定 load_all 模板。
    code, result = "", None
    last_error = ""
    # 【数据源约束】逐文件列出真实键名和 sheet 清单，多 sheet 文件明确说明
    # "月度数据在 sheet 里"，堵住模型臆造分月文件名的最大错误源
    sheets_map = _real_sheets(data_files)
    allowed_lines = []
    for f in data_files:
        name = f.split("/")[-1]
        sheets = sheets_map.get(name)
        if sheets and len(sheets) > 1:
            allowed_lines.append(
                f'DATA_FILES["{name}"]（该文件共 {len(sheets)} 个 sheet：{"、".join(sheets)}。'
                "月度数据全部在这些 sheet 里，必须用 load_all 一次读入全部，"
                '用"数据来源sheet"列区分月份筛选，绝对禁止臆造"文件名_月份.xlsx"之类的分月文件）'
            )
        else:
            allowed_lines.append(f'DATA_FILES["{name}"]')
    allowed = "\n".join(allowed_lines) + "\n（以上为本会话全部已登记文件，键名逐字照抄，不存在其他文件）"
    for attempt in range(max_attempts):
        temperature = 0.1 if attempt == 0 else 0.3
        llm = model_router.chat_model_for("code", temperature=temperature)
        prompt = CODE_GEN_PROMPT.format(schema=schema, explore=explore, columns=columns, question=question, n_files=len(data_files),
                                        allowed=allowed)
        if last_error:
            prompt += (
                "\n\n【上一次生成的脚本执行报错，请修正后重新输出完整脚本。"
                "检查是否使用了不存在的列名或表结构理解错误。】\n"
                f"报错信息：\n{last_error[:1500]}"
                + _column_hint(last_error, explore)
                + "\n注意：仍必须以上一轮同样的方式开头——报表型文件一律用 "
                  "df = load_all(DATA_FILES[\"文件名\"])，禁止改回 pd.read_excel 直接读。"
            )
        raw = llm.invoke(prompt).content
        code = _fix_imports(_extract_code(raw))
        code = _repair_column_names(code, explore + "\n" + schema)
        code = _repair_file_keys(code, data_files)
        code = _repair_read_calls(code, data_files)
        code = _repair_sheet_names(code, data_files)
        logger.info("generated code (attempt %d):\n%s", attempt + 1, code)
        result = get_executor().execute(code, files=data_files, timeout=timeout)
        if result.success:
            break
        # 去掉 PerformanceWarning 等噪音，报错反馈的 1500 字预算留给真正有用的信息
        clean_err = "\n".join(
            ln for ln in result.stderr.splitlines()
            if "Warning" not in ln and "sandbox_" not in ln
        )
        last_error = ((result.error or "") + "\n" + clean_err).strip() or "未知错误"
        logger.warning("execution failed (attempt %d): %s", attempt + 1, last_error[:800])

    assert result is not None

    advice = ""
    if result.success and result.stdout.strip():
        advisor = model_router.chat_model_for("chat", temperature=0.4)
        advice = advisor.invoke(ADVICE_PROMPT.format(
            question=question,
            stdout=result.stdout[-4000:],
            stderr=result.stderr[-1500:],
        )).content
    elif result.success:
        advice = "脚本执行成功但没有输出。请补充更明确的分析指标或检查数据内容。"
    else:
        err_tail = last_error[-600:]
        advice = (
            f"分析脚本执行失败：{result.error or result.stderr[:300]}。\n\n"
            f"报错摘要：\n```\n{err_tail}\n```\n\n"
            "建议：检查数据文件是否与问题匹配、表述更明确（如指出用哪张表的哪一列），或补充数据后重试。"
        )

    return {
        "intent": "analysis",
        "answer": advice,
        "code": code,
        "stdout": result.stdout,
        "stderr": result.stderr[:1000] if not result.success else "",
        "success": result.success,
        "images": result.images,  # base64 PNG 列表
    }
