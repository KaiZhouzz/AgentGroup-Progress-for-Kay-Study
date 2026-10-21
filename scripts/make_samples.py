"""生成样例数据（可重复运行，覆盖旧文件）：

- data/samples/billings_2025H1.xlsx：约 200 行账单（房间号、月份、应收、实收、欠费）
- data/samples/运营管理制度.txt：水电费催缴流程、退租结算流程等制度段落
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
random.seed(42)

ROOMS = [f"{b}{f:02d}" for b in ["A", "B", "C"] for f in range(1, 15)]  # 42 间房
MONTHS = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]


def make_billings() -> Path:
    rows = []
    for month in MONTHS:
        for room in ROOMS:
            rent = random.choice([1800, 2000, 2200, 2500])
            utilities = round(random.uniform(50, 300), 2)
            receivable = round(rent + utilities, 2)
            # 约 12% 的房间欠费，欠费金额随机
            if random.random() < 0.12:
                unpaid = round(random.uniform(100, receivable), 2)
            else:
                unpaid = 0.0
            rows.append({
                "房间号": room,
                "月份": month,
                "应收": receivable,
                "实收": round(receivable - unpaid, 2),
                "欠费": unpaid,
            })
    df = pd.DataFrame(rows)
    out = SAMPLES / "billings_2025H1.xlsx"
    df.to_excel(out, index=False)
    return out


POLICY_TEXT = """长租公寓运营管理制度（节选）

一、水电费催缴流程
1. 每月 5 日前，管家完成各房间水电表读数抄录并录入系统生成账单。
2. 每月 8 日通过 APP 与短信向租客推送账单，账单需在每月 15 日前缴纳。
3. 逾期未缴的，第 16 日起每日按欠费金额的 0.05% 计收滞纳金，并发送第一次催缴通知。
4. 逾期超过 7 天的，管家上门沟通并送达书面催缴单，同时暂停公共区域增值服务。
5. 逾期超过 15 天仍未缴纳的，按租赁合同第 8.3 条处理，并纳入租客信用记录。

二、退租结算流程
1. 租客需提前 30 天在 APP 提交退租申请，管家 24 小时内确认并预约验房时间。
2. 退租当日，管家与租客共同验房，核对家具家电完好情况并拍照存档。
3. 结算内容包含：未结租金、水电费、物品损坏赔偿、卫生清洁费（如适用）。
4. 押金在扣除应扣款项后，于验房完成后 7 个工作日内原路退回。
5. 如租客存在违约情形，按合同约定扣除违约金后退还剩余押金，并向租客出具结算清单。

三、维修响应规范
1. 紧急维修（漏水、断电、门锁损坏）2 小时内响应，24 小时内修复。
2. 一般维修 24 小时内响应，3 个工作日内修复。
3. 维修完成后由租客在 APP 确认，管家回访满意度。

四、公共空间管理
1. 公共厨房、健身房使用时间为每日 6:00-23:00。
2. 禁止在公共区域堆放私人物品超过 48 小时。
3. 社区活动每月至少举办 2 场，由运营专员负责策划与报名组织。
"""


def make_policy() -> Path:
    out = SAMPLES / "运营管理制度.txt"
    out.write_text(POLICY_TEXT, encoding="utf-8")
    return out


if __name__ == "__main__":
    SAMPLES.mkdir(parents=True, exist_ok=True)
    print("生成:", make_billings())
    print("生成:", make_policy())
