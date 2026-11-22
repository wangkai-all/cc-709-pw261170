"""以连通组为最小单位的集合分配。

输入是已经完成时间评价的组。分配过程分两层：

1. **强制层**：被时间约束 shift 的组直接进入最早可去集合；隔离组不参与；
2. **自由层**：剩余组按"大组优先"（LPT 风格）依次放入当前缺口最大的集合，
   大组因此会主导最终比例——这正是页面要解释的偏差来源。

组规模相同、缺口相同时，用由种子派生的确定性哈希打破平局，
所以同样的输入 + 种子永远产出同样的分配。
组数量不超过 ``exhaustive_limit`` 时，对自由组做小规模穷举，
取误差平方和（SSE）最小的方案，并与贪心结果一起返回，供测试与页面说明。
"""
from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field

from .graph import Group
from .timesplit import SPLITS, TimeEvaluation


@dataclass
class GroupStat:
    group: Group
    sample_count: int
    subject_count: int
    eligible_splits: tuple[str, ...]
    forced_split: str | None
    crosses_boundary: bool
    min_date: str
    max_date: str


@dataclass
class AllocationResult:
    # group_id -> split
    assignment: dict[str, str]
    # group_id -> 人类可读的分配理由（页面逐条展示）
    reasons: dict[str, str]
    # 每个集合的样本数 / 主体数
    split_samples: dict[str, int]
    split_subjects: dict[str, int]
    target_ratios: dict[str, float]
    actual_ratios: dict[str, float]
    # 目标样本量与实际样本量、绝对偏差
    target_samples: dict[str, float]
    deviation: dict[str, float]
    sse: float
    strategy: str  # "exhaustive" 或 "greedy"
    exhaustive_limit: int
    greedy_sse: float | None = None  # 穷举模式下同时记录贪心 SSE
    notes: list[str] = field(default_factory=list)


def _seeded_hash(seed: int, *parts: str) -> str:
    h = hashlib.sha256()
    h.update(str(seed).encode("utf-8"))
    for p in parts:
        h.update(b"|")
        h.update(p.encode("utf-8"))
    return h.hexdigest()


def _sse(actual: dict[str, int], target: dict[str, float]) -> float:
    return sum((actual[sp] - target[sp]) ** 2 for sp in SPLITS)


def _ratios(counts: dict[str, int], total: int) -> dict[str, float]:
    if total == 0:
        return {sp: 0.0 for sp in SPLITS}
    return {sp: counts[sp] / total for sp in SPLITS}


def normalize_ratios(raw: dict[str, float]) -> dict[str, float]:
    """把用户输入的比例归一化到和为 1；全 0 等分给三份。"""
    picked = {sp: max(0.0, float(raw.get(sp, 0.0))) for sp in SPLITS}
    total = sum(picked.values())
    if total <= 0:
        return {sp: 1 / 3 for sp in SPLITS}
    return {sp: picked[sp] / total for sp in SPLITS}


def _build_stats(
    groups: list[Group],
    samples_per_group: dict[str, int],
    time_eval: TimeEvaluation,
) -> list[GroupStat]:
    stats: list[GroupStat] = []
    for g in groups:
        info = time_eval.infos[g.group_id]
        stats.append(
            GroupStat(
                group=g,
                sample_count=samples_per_group[g.group_id],
                subject_count=g.subject_count,
                eligible_splits=tuple(sp for sp in SPLITS if info.eligible[sp]),
                forced_split=time_eval.forced_split.get(g.group_id),
                crosses_boundary=info.crosses_boundary,
                min_date=info.min_date,
                max_date=info.max_date,
            )
        )
    return stats


def _tie_break_key(seed: int, stat: GroupStat) -> str:
    return _seeded_hash(seed, stat.group.group_id)


def _greedy(
    free: list[GroupStat],
    counts: dict[str, int],
    target: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """大组优先 + 缺口最大优先的确定性贪心。"""
    assignment: dict[str, str] = {}
    # 大组先放；规模相同按种子哈希定序，保证可复现
    ordered = sorted(
        free,
        key=lambda st: (
            -st.sample_count,
            _tie_break_key(seed, st),
        ),
    )
    for st in ordered:
        candidates = [sp for sp in st.eligible_splits]
        # 缺口 = 目标量 - 当前量；越大越该补
        best = max(
            candidates,
            key=lambda sp: (
                target[sp] - counts[sp],
                # 平局时按固定集合顺序，再按种子哈希，彻底消除不确定性
                -SPLITS.index(sp),
                _seeded_hash(seed, st.group.group_id, sp),
            ),
        )
        assignment[st.group.group_id] = best
        counts[best] += st.sample_count
    return assignment


def _exhaustive(
    free: list[GroupStat],
    base_counts: dict[str, int],
    target: dict[str, float],
    seed: int,
) -> tuple[dict[str, str], float]:
    """枚举自由组的全部可行分配，取 SSE 最小且平局确定性的方案。"""
    best_assignment: dict[str, str] | None = None
    best_sse = float("inf")
    best_key: tuple[int, ...] | None = None

    domains = [list(st.eligible_splits) for st in free]
    if not domains:
        return {}, _sse(dict(base_counts), target)
    for combo in itertools.product(*domains):
        counts = dict(base_counts)
        for st, sp in zip(free, combo):
            counts[sp] += st.sample_count
        sse = _sse(counts, target)
        # 平局用“各集合样本数元组”和种子哈希做确定性选择
        tie_key = tuple(counts[sp] for sp in SPLITS)
        if sse < best_sse or (
            sse == best_sse
            and (
                best_key is None
                or tie_key < best_key
                or (
                    tie_key == best_key
                    and _seeded_hash(seed, *combo)
                    < _seeded_hash(seed, *(best_assignment or {}).values())
                )
            )
        ):
            best_sse = sse
            best_key = tie_key
            best_assignment = {
                st.group.group_id: sp for st, sp in zip(free, combo)
            }
    assert best_assignment is not None
    return best_assignment, best_sse


def allocate(
    groups: list[Group],
    samples_per_group: dict[str, int],
    time_eval: TimeEvaluation,
    target_ratios: dict[str, float],
    seed: int,
    exhaustive_limit: int = 10,
) -> AllocationResult:
    """执行组级分配并产出解释信息。"""
    stats = _build_stats(groups, samples_per_group, time_eval)
    total_samples = sum(st.sample_count for st in stats)
    quarantined_count = sum(
        st.sample_count
        for st in stats
        if st.group.group_id in time_eval.quarantined
    )
    allocatable = total_samples - quarantined_count
    ratios = normalize_ratios(target_ratios)
    target_samples = {sp: ratios[sp] * allocatable for sp in SPLITS}

    counts = {sp: 0 for sp in SPLITS}
    subject_counts = {sp: 0 for sp in SPLITS}
    assignment: dict[str, str] = {}
    reasons: dict[str, str] = {}

    # ---- 第一层：时间强制（shift） ----
    forced_stats = [st for st in stats if st.forced_split is not None]
    free_stats = [
        st
        for st in stats
        if st.forced_split is None
        and st.group.group_id not in time_eval.quarantined
    ]
    for st in sorted(forced_stats, key=lambda x: (-x.sample_count, x.group.group_id)):
        sp = st.forced_split
        assert sp is not None
        assignment[st.group.group_id] = sp
        counts[sp] += st.sample_count
        subject_counts[sp] += st.subject_count
        reasons[st.group.group_id] = (
            f"时间约束优先：组内最晚样本 {st.max_date} 不满足更早集合，"
            f"按 shift 规则整组后移到 {sp}，不参与比例分配"
        )

    # ---- 第二层：自由组（贪心，必要时穷举） ----
    for st in free_stats:
        if not st.eligible_splits:
            raise RuntimeError(
                f"组 {st.group.group_id} 没有任何时间合格的集合，"
                f"应在时间评价阶段被隔离，请检查 isolate 规则配置"
            )
    base_counts = dict(counts)
    use_exhaustive = len(free_stats) <= exhaustive_limit
    if use_exhaustive:
        greedy_assignment = _greedy(free_stats, dict(base_counts), target_samples, seed)
        greedy_counts = dict(base_counts)
        for st in free_stats:
            greedy_counts[greedy_assignment[st.group.group_id]] += st.sample_count
        greedy_sse = _sse(greedy_counts, target_samples)

        chosen, chosen_sse = _exhaustive(
            free_stats, base_counts, target_samples, seed
        )
        strategy = "exhaustive"
    else:
        chosen = _greedy(free_stats, dict(base_counts), target_samples, seed)
        greedy_sse = None
        final_counts = dict(base_counts)
        for st in free_stats:
            final_counts[chosen[st.group.group_id]] += st.sample_count
        chosen_sse = _sse(final_counts, target_samples)
        strategy = "greedy"

    # 无论哪条路径，都要把自由组的选择并入最终分配与计数
    for st in free_stats:
        sp = chosen[st.group.group_id]
        assignment[st.group.group_id] = sp
        counts[sp] += st.sample_count
        subject_counts[sp] += st.subject_count

    # 自由组的解释：按分配顺序（大组优先）说明它如何改变比例
    ordered_free = sorted(
        free_stats,
        key=lambda st: (-st.sample_count, _tie_break_key(seed, st)),
    )
    running = dict(base_counts)
    for st in ordered_free:
        sp = chosen[st.group.group_id]
        gap_before = {k: target_samples[k] - running[k] for k in SPLITS}
        running[sp] += st.sample_count
        allowed = "/".join(st.eligible_splits)
        reasons[st.group.group_id] = (
            f"大组优先排序后处理（本组 {st.sample_count} 个样本）；"
            f"时间允许集合：{allowed}；分配前各集合缺口 "
            f"train={gap_before['train']:.1f}, val={gap_before['val']:.1f}, "
            f"test={gap_before['test']:.1f}，缺口最大的 {sp} 获配整组"
        )

    for gid, reason in time_eval.quarantined.items():
        reasons[gid] = f"隔离：{reason}"

    actual = _ratios(counts, allocatable)
    deviation = {sp: counts[sp] - target_samples[sp] for sp in SPLITS}

    notes: list[str] = []
    largest = max((st.sample_count for st in stats), default=0)
    if allocatable and largest / allocatable >= 0.25:
        big = next(st for st in stats if st.sample_count == largest)
        notes.append(
            f"最大连通组 {big.group.group_id} 含 {largest} 个样本"
            f"（占可分配样本 {largest / allocatable:.1%}），必须整体放入一个集合，"
            f"任何集合接收它都会至少偏离目标 {largest / 2:.0f} 个样本量级，"
            f"这是比例无法精确命中的结构性原因"
        )
    if forced_stats:
        notes.append(
            f"{len(forced_stats)} 个组被时间约束强制后移，未参与比例优化，"
            f"进一步压缩了可调空间"
        )
    if time_eval.quarantined:
        notes.append(
            f"{len(time_eval.quarantined)} 个组按 isolate 规则隔离，"
            f"共 {quarantined_count} 个样本不进入任何目标集合"
        )
    if strategy == "exhaustive" and greedy_sse is not None and greedy_sse > chosen_sse:
        notes.append(
            f"自由组仅 {len(free_stats)} 个，已穷举全部可行方案；"
            f"最优 SSE={chosen_sse:.1f}，贪心 SSE={greedy_sse:.1f}"
        )
    elif strategy == "exhaustive":
        notes.append(
            f"自由组仅 {len(free_stats)} 个，已穷举全部可行方案，"
            f"贪心恰好也是最优（SSE={chosen_sse:.1f}）"
        )
    else:
        notes.append(
            f"自由组 {len(free_stats)} 个超过穷举上限 {exhaustive_limit}，"
            f"使用确定性贪心（大组优先 + 最大缺口）"
        )

    return AllocationResult(
        assignment=assignment,
        reasons=reasons,
        split_samples=dict(counts),
        split_subjects=subject_counts,
        target_ratios=ratios,
        actual_ratios=actual,
        target_samples=target_samples,
        deviation=deviation,
        sse=chosen_sse,
        strategy=strategy,
        exhaustive_limit=exhaustive_limit,
        greedy_sse=greedy_sse,
        notes=notes,
    )
