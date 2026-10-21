"""验证 PLAN 阶段（语义质疑）：用有歧义的需求跑流水线，看 14B 是否会反问口径。

用法: python scripts/test_planner.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.pipeline import AnalysisPipeline

DATA_FILES = [
    str(Path("data/samples/rooms_info.xlsx").resolve()),
    str(Path("data/samples/billings_2025H1.xlsx").resolve()),
]

# 刻意模糊：跨两份表、没指定关联键、没指定欠费口径 → 理想情况下 PLAN 应反问
QUESTION = "帮我分析一下欠费情况"

pipeline = AnalysisPipeline(DATA_FILES)
result = pipeline.start(QUESTION)

print("=" * 60)
print("intent:", result.get("intent"))
print("success:", result.get("success"))
print("-" * 60)
if result.get("intent") == "clarify_analysis":
    print("PLAN 阶段提出了澄清问题（质疑生效）:")
    for q in result.get("questions", []):
        print(f"\n[{q.get('id')}] {q.get('question')}")
        for o in q.get("options", []):
            print(f"    - {o}")
else:
    print("PLAN 未提问，直接执行了。answer 前 500 字:")
    print((result.get("answer") or "")[:500])
