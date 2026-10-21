"""文件解析：把 pdf/docx/xlsx/txt/csv 解析为文本块（RAG 入库）或数据概要（分析用）。"""
from __future__ import annotations

from pathlib import Path

TEXT_EXTS = {".txt", ".md"}
EXCEL_EXTS = {".xlsx", ".xls", ".csv"}
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}
SUPPORTED_EXTS = TEXT_EXTS | EXCEL_EXTS | PDF_EXTS | DOCX_EXTS


def load_document(path: str | Path) -> list[str]:
    """把文档解析成文本块列表。xlsx 返回每个 sheet 的文本化表示。"""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in TEXT_EXTS:
        return _chunk_text(path.read_text(encoding="utf-8", errors="ignore"))
    if ext in DOCX_EXTS:
        return _load_docx(path)
    if ext in PDF_EXTS:
        return _load_pdf(path)
    if ext in EXCEL_EXTS:
        return _load_excel_as_text(path)
    raise ValueError(f"不支持的文件类型: {ext}")


def parse_dataframe_summary(path: str | Path) -> str:
    """分析用：读取表格文件，输出列名、类型、行数、样例行、缺失率概要。

    针对"报表型"Excel（合并单元格、多层表头、Unnamed 列）做清洗：
    Unnamed 列用前几行的实际表头值补齐，全空列跳过，避免概要里充斥 Unnamed: N。
    """
    import pandas as pd

    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        sheets = {"(csv)": df}
    else:
        sheets = pd.read_excel(path, sheet_name=None)

    parts = [f"文件: {path.name}"]
    first_cols: list[str] | None = None
    for name, df in sheets.items():
        head = df.head(3)
        cleaned = []
        for col in df.columns:
            if df[col].notna().sum() == 0:
                continue  # 整列空值，跳过
            if isinstance(col, str) and col.startswith("Unnamed"):
                real = next(
                    (str(head.iloc[r][col]) for r in range(len(head)) if pd.notna(head.iloc[r][col])),
                    None,
                )
                cleaned.append(real if real else str(col))
            else:
                cleaned.append(str(col))
        if first_cols is None:
            first_cols = cleaned
            parts.append(f"\n== Sheet [{name}] 共 {len(df)} 行 x {len(df.columns)} 列 ==")
            parts.append("列(清洗后): " + ", ".join(cleaned))
            parts.append("样例前2行(截断):")
            sample = df.dropna(how="all").head(2)
            parts.append(sample.to_string(index=False, max_cols=10)[:600])
        elif cleaned == first_cols:
            # 列结构与首个 sheet 相同（按 sheet 分月的报表很常见），只留行数
            parts.append(f"== Sheet [{name}] 共 {len(df)} 行（列同首个 sheet）")
        else:
            parts.append(f"== Sheet [{name}] 共 {len(df)} 行（列不同）: " + ", ".join(cleaned)[:400])
    summary = "\n".join(parts)
    if len(summary) > 6000:
        summary = summary[:6000] + "\n...（概要过长已截断）"
    return summary


def _chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    """按段落聚合 + 长度上限切块。"""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 1 > max_chars and buf:
            chunks.append(buf)
            buf = buf[-overlap:] + "\n" + p if overlap else p
        else:
            buf = f"{buf}\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]


def _load_docx(path: Path) -> list[str]:
    import docx

    doc = docx.Document(str(path))
    text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    # 表格内容也纳入
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                text += "\n" + " | ".join(cells)
    return _chunk_text(text)


def _load_pdf(path: Path) -> list[str]:
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                pages.append(t.strip())
    if pages:
        return _chunk_text("\n\n".join(pages))
    # 扫描件（无文字层）：渲染成图片后走 OCR，按"页"切块——
    # 表格型文档同一页的行互为上下文，切碎会让"品名↔数量"配对丢失
    return [p for p in _ocr_pdf_pages(path) if p.strip()]


def _ocr_pdf_pages(path: Path, dpi: int = 150) -> list[str]:
    """用 PyMuPDF 渲染 + RapidOCR（中文）识别扫描版 PDF，每页返回一个文本块。"""
    import pymupdf as fitz
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR

    ocr = RapidOCR()
    pages: list[str] = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:  # RGBA -> RGB
                img = img[:, :, :3]
            result, _ = ocr(img)
            lines = _rows_from_ocr(result) if result else []
            if lines:
                pages.append("\n".join(_label_table_rows(lines)))
    return pages


# 表格表头里常见列名；命中即认为该页是表格，用表头给后续数据行打标签
_HEADER_KEYS = ("品名", "数量", "单价", "金额", "单位")


def _label_table_rows(lines: list[str]) -> list[str]:
    """把按位置排列的 "a | b | c" 行转成 "列名=值" 的自描述格式。

    纯位置对应（第3列=数量）对 LLM 太隐晦，极易读串列（把金额当数量）；
    键值对格式让每行的语义显式化。
    """
    out: list[str] = []
    header: list[str] | None = None
    for line in lines:
        cells = [c.strip() for c in line.split("|")]
        is_header = len(cells) >= 3 and sum(1 for c in cells if any(k in c for k in _HEADER_KEYS)) >= 2
        if is_header:
            header = cells
            out.append(line)
            continue
        if header and len(cells) == len(header):
            out.append("，".join(f"{h}={c}" for h, c in zip(header, cells) if h and c))
        elif header and len(cells) == len(header) - 1:
            # 数据行缺了"序号"列（空单元格 OCR 识别不到），右对齐后打标签
            out.append("，".join(f"{h}={c}" for h, c in zip(header[1:], cells) if h and c))
        else:
            out.append(line)
    return out


def _rows_from_ocr(result: list, y_tol: float = 12.0) -> list[str]:
    """把 OCR 检测框按纵坐标聚成行、行内按横坐标排序，恢复表格的行列结构。

    RapidOCR 原始输出按检测顺序排列，逐行打印时同一列的格子会互相穿插，
    导致"数量/单价"错位；按坐标重建后每行变成 "品名 | 单位 | 数量 | 单价 | 金额"。
    """
    items = []
    for box, text, _score in result:
        yc = sum(p[1] for p in box) / 4.0
        xc = sum(p[0] for p in box) / 4.0
        items.append((yc, xc, text.strip()))
    items.sort(key=lambda t: (t[0], t[1]))

    rows: list[list[tuple[float, str]]] = []
    row_ys: list[float] = []
    for yc, xc, text in items:
        if not text:
            continue
        if rows and abs(yc - row_ys[-1]) <= y_tol:
            rows[-1].append((xc, text))
            row_ys[-1] = (row_ys[-1] * (len(rows[-1]) - 1) + yc) / len(rows[-1])
        else:
            rows.append([(xc, text)])
            row_ys.append(yc)
    return [" | ".join(t for _, t in sorted(r, key=lambda x: x[0])) for r in rows]


def _load_excel_as_text(path: Path) -> list[str]:
    """xlsx 入库 RAG 时的文本化：每个 sheet 的行记录拼成文本后切块。"""
    import pandas as pd

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        sheets = {path.stem: df}
    else:
        sheets = pd.read_excel(path, sheet_name=None)

    lines: list[str] = []
    for name, df in sheets.items():
        lines.append(f"[Sheet: {name}]")
        lines.append("列名: " + ", ".join(map(str, df.columns)))
        for _, row in df.iterrows():
            lines.append("; ".join(f"{c}={v}" for c, v in row.items()))
    return _chunk_text("\n".join(lines))
