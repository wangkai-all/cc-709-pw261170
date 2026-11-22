#!/usr/bin/env python3
"""独立小规模穷举校验器。

不调用生产分配器的搜索代码，而是：

1. 构造若干小规模场景（含连通组、跨时间边界、隔离规则）；
2. 暴力枚举每个连通组到 train/val/test 的**全部**可行映射（时间不合格的去向直接剪枝）；
3. 对每张映射检查硬约束，并计算 SSE；
4. 断言生产管线的输出在所有枚举方案中 SSE 最小、且满足全部硬约束。

场景规模控制在 8 个组以内（3^8 = 6561 种映射，秒级完成）。

用法：python examples/verify.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from splitter.allocate import normalize_ratios  # noqa: E402
from splitter.dataset import Relation, Sample  # noqa: E402
from splitter.graph import build_groups  # noqa: E402
from splitter.manifest import SplitConfig, generate_manifests  # noqa: E402
from splitter.timesplit import SPLITS, TimePolicy, evaluate_groups  # noqa: E402

RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def make_dataset():
    """8 个组以内的小型场景，覆盖：二主体组、跨边界组、普通独立组。"""
    samples = []
    plan = {
        # subject: (dates, )
        "G1a": ("2025-02-01", "2025-04-01"),          # 与 G1b 连通
        "G1b": ("2025-03-01",),
        "G2a": ("2025-01-01", "2025-11-01"),          # 跨 train 边界，强制后移
        "G3": ("2025-02-15",),
        "G4": ("2025-05-10",),
        "G5": ("2025-06-30",),                         # 截止日当天
        "G6": ("2025-08-20",),
        "G7": ("2025-03-03",),
    }
    sid = 0
    for subject, dates in plan.items():
        for day in dates:
            sid += 1
            samples.append(Sample(f"s{sid:02d}", subject, day, {"k": str(sid)}))
    relations = [Relation("e1", "G1a", "G1b", "household")]
    from splitter.dataset import Dataset
    return Dataset("verify_tiny", samples, relations, "samples-v1", "relations-v1")


def enumerate_feasible(groups, time_eval):
    """生成全部满足时间资格的组->集合映射（隔离组不参与）。"""
    active = [g for g in groups if g.group_id not in time_eval.quarantined]
    domains = []
    for g in active:
        info = time_eval.infos[g.group_id]
        domains.append([sp for sp in SPLITS if info.eligible[sp]])
    for combo in itertools.product(*domains):
        yield dict(zip((g.group_id for g in active), combo))


def check_hard_constraints(assignment, groups, samples, time_eval):
    """返回 (是否通过, 违规原因列表)。"""
    violations = []
    # 1. 每个非隔离组恰好一个去向，且时间合格
    for g in groups:
        if g.group_id in time_eval.quarantined:
            if g.group_id in assignment:
                violations.append(f"隔离组 {g.group_id} 不应被分配")
            continue
        if g.group_id not in assignment:
            violations.append(f"组 {g.group_id} 缺少去向")
            continue
        sp = assignment[g.group_id]
        if not time_eval.infos[g.group_id].eligible[sp]:
            violations.append(f"组 {g.group_id} 进入时间不合格集合 {sp}")
    # 2. 同组主体不跨集合（组级分配天然保证，这里独立复核样本归属）
    subject_split = {}
    group_of = {sub: g.group_id for g in groups for sub in g.subjects}
    for sample in samples:
        gid = group_of[sample.subject_id]
        if gid in time_eval.quarantined:
            continue
        sp = assignment.get(gid)
        if subject_split.setdefault(sample.subject_id, sp) != sp:
            violations.append(f"主体 {sample.subject_id} 的样本跨集合")
    # 3. 截止日后的样本绝不在 train
    for sample in samples:
        gid = group_of[sample.subject_id]
        if assignment.get(gid) == "train" and sample.event_date > "2025-06-30":
            violations.append(f"迟到样本 {sample.sample_id} 流入 train")
        if assignment.get(gid) == "val" and sample.event_date > "2025-09-30":
            violations.append(f"迟到样本 {sample.sample_id} 流入 val")
    return not violations, violations


def sse_of(assignment, groups, samples_per_group, target_samples):
    counts = {sp: 0 for sp in SPLITS}
    for gid, sp in assignment.items():
        counts[sp] += samples_per_group[gid]
    return sum((counts[sp] - target_samples[sp]) ** 2 for sp in SPLITS), counts


def run(rule: str):
    dataset = make_dataset()
    config = SplitConfig(
        seed=20261122,
        target_ratios=RATIOS,
        train_cutoff="2025-06-30",
        val_cutoff="2025-09-30",
        cross_boundary_rule=rule,
    )
    groups, skipped = build_groups(dataset.samples, dataset.relations)
    assert not skipped
    policy = TimePolicy(config.train_cutoff, config.val_cutoff, rule)
    time_eval = evaluate_groups(groups, dataset.samples, policy)
    spg = {
        g.group_id: sum(1 for s in dataset.samples if s.subject_id in set(g.subjects))
        for g in groups
    }
    allocatable = sum(
        n for gid, n in spg.items() if gid not in time_eval.quarantined
    )
    target = {sp: normalize_ratios(RATIOS)[sp] * allocatable for sp in SPLITS}

    feasible = list(enumerate_feasible(groups, time_eval))
    print(f"[rule={rule}] 组数={len(groups)} 枚举可行映射={len(feasible)} "
          f"隔离组={len(time_eval.quarantined)} 强制组={len(time_eval.forced_split)}")

    best_sse = float("inf")
    best_maps = []
    checked = 0
    for assignment in feasible:
        ok, why = check_hard_constraints(assignment, groups, dataset.samples, time_eval)
        if not ok:
            raise AssertionError(f"枚举到违规映射: {why}")
        checked += 1
        sse, counts = sse_of(assignment, groups, spg, target)
        if sse < best_sse - 1e-9:
            best_sse, best_maps = sse, [assignment]
        elif abs(sse - best_sse) <= 1e-9:
            best_maps.append(assignment)
    print(f"[rule={rule}] 全部 {checked} 张映射通过硬约束；暴力最优 SSE={best_sse:.4f}")

    # 生产管线结果
    gen = generate_manifests(dataset, config)
    prod = {g["group_id"]: g["assigned_split"] for g in gen.groups
            if g["assigned_split"] is not None}
    prod_sse = gen.allocation["sse"]
    ok, why = check_hard_constraints(prod, groups, dataset.samples, time_eval)
    assert ok, f"生产输出违规: {why}"
    assert abs(prod_sse - best_sse) <= 1e-6, (
        f"生产 SSE={prod_sse} 不是枚举最优 {best_sse}"
    )
    assert prod in best_maps, "生产分配不在最优方案集合内"
    print(f"[rule={rule}] 生产管线 SSE={prod_sse:.4f}，与暴力最优一致 ✔")
    print(f"[rule={rule}] 实际比例 "
          + ", ".join(f"{sp}={gen.allocation['actual_ratios'][sp]:.2%}" for sp in SPLITS))


if __name__ == "__main__":
    run("shift")
    run("isolate")
    print("\n全部小规模穷举校验通过。")
