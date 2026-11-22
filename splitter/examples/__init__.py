"""内置示例：一个很大的关联组、间接三层关系、跨时间边界与迟到关系修订。

数据布局（24 个主体 / 72 个样本，日期跨度 2025-01 ~ 2025-12）：

1. **超大关联组** ``mega``：M01..M10 经 9 条 ``household`` 边连成 10 主体组，
   30 个样本（占总量 41.7%），且组内最晚样本 2025-11-18 越过两个时间边界，
   只能整体进 test——这是比例偏离目标的结构性原因。
2. **间接三层关系链** ``chain``：C01 -family-> C02 -household-> C03
   -same_address-> C04，C01 与 C04 没有直接边，但经三条边（三跳）间接连通，
   必须同组；C04 的 12 月样本让整组越过 val 边界，演示 shift / isolate。
3. **迟到关系修订**：初版发布后才发现 C04 与独立主体 S01 实际是同一人，
   补一条 ``same_as`` 边会把 test 里的 chain 组与 train 里的 S01 合并，
   形成跨集合冲突（潜在泄漏）。
4. 其余独立主体 S01..S10 用于比例分配与泄漏对照。
"""
from __future__ import annotations

from ..dataset import Dataset, Relation, Sample

TRAIN_CUTOFF = "2025-06-30"
VAL_CUTOFF = "2025-09-30"


def _sample(sid: str, subject: str, day: str, amount: str) -> Sample:
    return Sample(
        sample_id=sid,
        subject_id=subject,
        event_date=day,
        attributes={"amount": amount, "channel": "web"},
    )


def build_example_dataset() -> Dataset:
    samples: list[Sample] = []

    # ---- 超大关联组 M01..M10，每主体 3 个样本，共 30 个 ----
    mega_dates = ["2025-02-10", "2025-05-22", "2025-11-18"]
    for i in range(1, 11):
        subject = f"M{i:02d}"
        for j, day in enumerate(mega_dates, start=1):
            samples.append(
                _sample(f"s_{subject}_{j}", subject, day, str(100 + i * 10 + j))
            )

    # ---- 三层间接关系链 C01 -> C02 -> C03 -> C04 ----
    chain_dates = {
        "C01": ["2025-01-15", "2025-04-20", "2025-07-10"],
        "C02": ["2025-03-05", "2025-08-18"],
        "C03": ["2025-02-28", "2025-09-12"],
        # C04 的 12 月样本使整组越过 val_cutoff
        "C04": ["2025-05-09", "2025-12-03"],
    }
    for subject, days in chain_dates.items():
        for j, day in enumerate(days, start=1):
            samples.append(
                _sample(f"s_{subject}_{j}", subject, day, str(200 + j))
            )

    # ---- 独立主体 S01..S10，各 3 个样本，共 30 个 ----
    # S01..S05 全部落在 train 时间窗内（自由组，供比例分配调节）；
    # S06..S10 越过不同边界，制造强制后移与跨边界场景。
    solo_windows = [
        ("2025-01-08", "2025-03-19", "2025-06-11"),   # 全在 train 窗
        ("2025-01-14", "2025-04-02", "2025-06-25"),   # 全在 train 窗
        ("2025-02-03", "2025-03-27", "2025-05-16"),   # 全在 train 窗
        ("2025-02-18", "2025-04-29", "2025-06-07"),   # 全在 train 窗
        ("2025-03-08", "2025-05-12", "2025-06-29"),   # 全在 train 窗（含截止日当天）
        ("2025-02-14", "2025-05-30", "2025-08-25"),   # 跨 train/val，最早可去 val
        ("2025-03-22", "2025-07-15", "2025-10-08"),   # 跨 val/test，强制 test
        ("2025-01-30", "2025-04-17", "2025-09-02"),   # 越过 train 边界、未越 val，强制 val
        ("2025-02-23", "2025-06-08", "2025-09-22"),   # 同上，强制 val
        ("2025-03-11", "2025-08-02", "2025-12-15"),   # 跨 val/test，强制 test
    ]
    for i, days in enumerate(solo_windows, start=1):
        subject = f"S{i:02d}"
        for j, day in enumerate(days, start=1):
            samples.append(
                _sample(f"s_{subject}_{j}", subject, day, str(300 + i + j))
            )

    relations: list[Relation] = []

    # mega 组：9 条 household 边把 M01..M10 串起来（连通即可，无需全连接）
    for i in range(1, 10):
        relations.append(
            Relation(
                rel_id=f"rel_house_m{i:02d}_{i + 1:02d}",
                src_subject=f"M{i:02d}",
                dst_subject=f"M{i + 1:02d}",
                rel_type="household",
            )
        )

    # 三层间接关系：三种不同关系类型，强调"指定关系间接关联也算同组"
    relations.extend(
        [
            Relation("rel_c01_c02", "C01", "C02", "family"),
            Relation("rel_c02_c03", "C02", "C03", "household"),
            Relation("rel_c03_c04", "C03", "C04", "same_address"),
        ]
    )

    return Dataset(
        name="lake_demo",
        samples=samples,
        relations=relations,
        sample_version="samples-v1",
        relation_version="relations-v1",
    )


def example_config(
    cross_boundary_rule: str = "shift",
    exhaustive_limit: int = 10,
) -> dict:
    return {
        "seed": 20261122,
        "target_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "train_cutoff": TRAIN_CUTOFF,
        "val_cutoff": VAL_CUTOFF,
        "cross_boundary_rule": cross_boundary_rule,
        "relation_types": None,
        "exhaustive_limit": exhaustive_limit,
    }


LATE_REVISION = {
    "action": "merge",
    "subject_a": "C04",
    "subject_b": "S01",
    "rel_type": "same_as",
    "description": (
        "发布后人工核对发现 C04 与 S01 为同一人（迟到的关系修订），"
        "新增 same_as 边会把 test 中的 chain 组与 train 中的 S01 连通"
    ),
}
