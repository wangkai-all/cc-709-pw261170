"""时间约束：决定每个连通组允许进入哪些集合。

时间约束优先于比例目标。约定两个边界：

- ``train_cutoff``：训练截止日，晚于该日期的样本不能进入训练集；
- ``val_cutoff``：验证截止日，晚于该日期的样本不能进入验证集（测试集不设上界）。

一个组是整体评价的：组内**任意**样本超出某集合的时间窗，整个组都不能进该集合。
跨越时间边界的组（同时含边界前、后的样本）按规则处理，绝不拆组：

- ``shift``：整组后移到第一个时间上满足的集合；
- ``isolate``：整组隔离到 quarantine，不进入任何目标集合。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .dataset import Sample
from .graph import Group

SPLITS = ("train", "val", "test")
CROSS_RULES = ("shift", "isolate")


@dataclass
class TimePolicy:
    train_cutoff: str | None = None  # 含当天：event_date <= cutoff 才允许
    val_cutoff: str | None = None
    cross_boundary_rule: str = "shift"

    def __post_init__(self) -> None:
        if self.cross_boundary_rule not in CROSS_RULES:
            raise ValueError(
                f"cross_boundary_rule 必须是 {CROSS_RULES} 之一，"
                f"得到 {self.cross_boundary_rule!r}"
            )
        if self.train_cutoff is not None and self.val_cutoff is not None:
            if self.val_cutoff < self.train_cutoff:
                raise ValueError("val_cutoff 不能早于 train_cutoff")


@dataclass
class GroupTimeInfo:
    group_id: str
    min_date: str
    max_date: str
    # 各集合是否在时间上允许（test 永远允许）
    eligible: dict[str, bool]
    # 是否跨越 train/val 任一边界（组内同时存在边界前后的样本）
    crosses_boundary: bool
    # 被哪个边界限制（用于页面解释）
    limiting_cutoff: str | None
    sample_count: int


@dataclass
class TimeEvaluation:
    """所有组的时间评价，外加隔离清单与强制去向。"""

    infos: dict[str, GroupTimeInfo]
    # group_id -> 强制集合（shift 后最早可去集合）；隔离组不在此表中
    forced_split: dict[str, str] = field(default_factory=dict)
    # 隔离组及原因
    quarantined: dict[str, str] = field(default_factory=dict)
    missing_cutoff: bool = False


def _allowed_splits(min_date: str, max_date: str, policy: TimePolicy) -> tuple[
    dict[str, bool], bool, str | None
]:
    """根据组内最早/最晚样本日期计算各集合的时间资格。"""
    eligible = {"train": True, "val": True, "test": True}
    crosses = False
    limiting: str | None = None

    if policy.train_cutoff is not None:
        if max_date > policy.train_cutoff:
            eligible["train"] = False
            limiting = "train_cutoff"
            if min_date <= policy.train_cutoff:
                crosses = True
    if policy.val_cutoff is not None:
        if max_date > policy.val_cutoff:
            eligible["val"] = False
            if limiting is None:
                limiting = "val_cutoff"
            if min_date <= policy.val_cutoff:
                crosses = True
    return eligible, crosses, limiting


def evaluate_groups(
    groups: list[Group],
    samples: list[Sample],
    policy: TimePolicy,
) -> TimeEvaluation:
    """对每个连通组计算时间资格、强制去向或隔离。"""
    by_subject: dict[str, list[Sample]] = {}
    for s in samples:
        by_subject.setdefault(s.subject_id, []).append(s)

    infos: dict[str, GroupTimeInfo] = {}
    forced: dict[str, str] = {}
    quarantined: dict[str, str] = {}
    missing_cutoff = policy.train_cutoff is None and policy.val_cutoff is None

    for g in groups:
        group_samples = [
            s for subject in g.subjects for s in by_subject.get(subject, [])
        ]
        dates = sorted(s.event_date for s in group_samples)
        min_date, max_date = dates[0], dates[-1]
        eligible, crosses, limiting = _allowed_splits(min_date, max_date, policy)

        infos[g.group_id] = GroupTimeInfo(
            group_id=g.group_id,
            min_date=min_date,
            max_date=max_date,
            eligible=dict(eligible),
            crosses_boundary=crosses,
            limiting_cutoff=limiting,
            sample_count=len(group_samples),
        )

        if missing_cutoff:
            continue

        allowed = [sp for sp in SPLITS if eligible[sp]]
        if crosses and policy.cross_boundary_rule == "isolate":
            quarantined[g.group_id] = (
                f"组内含 {min_date}~{max_date} 样本，跨越时间边界"
                f"（{limiting}），按 isolate 规则整组隔离"
            )
            continue
        # shift：只有 train 不合格（组内含 train 截止日之后的样本）才强制
        # 进入最早可去集合（val 或 test）。由于 train_cutoff <= val_cutoff，
        # 资格是后缀链（train 合格必然 val/test 都合格），所以 train 合格的
        # 组一定是自由组，交给分配器按比例在三个集合间选择。
        if not eligible["train"]:
            forced[g.group_id] = allowed[0]

    return TimeEvaluation(
        infos=infos,
        forced_split=forced,
        quarantined=quarantined,
        missing_cutoff=missing_cutoff,
    )
