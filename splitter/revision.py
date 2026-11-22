"""关系修订：合并/撤回关系、潜在泄漏分析、新旧版本对比。

核心原则：**已发布清单不可变**。发现两组其实有关联时，旧清单原样保留，
只在其上标注潜在泄漏，并基于修订后的关系生成新候选版本，由用户确认发布。
"""
from __future__ import annotations

from dataclasses import replace

from .dataset import Dataset, Relation
from .graph import build_groups
from .manifest import Generation, SplitConfig, generate_manifests
from .timesplit import SPLITS


# ---------------------------------------------------------------------------
# 修订操作：返回新 Dataset，绝不原地修改
# ---------------------------------------------------------------------------

def withdraw_relation(dataset: Dataset, rel_id: str, note: str = "") -> Dataset:
    """撤回一条关系（active 置 false），关系版本递增。"""
    new_relations = []
    found = False
    for rel in dataset.relations:
        if rel.rel_id == rel_id:
            if not rel.active:
                raise ValueError(f"关系 {rel_id} 已是撤回状态")
            new_relations.append(replace(rel, active=False))
            found = True
        else:
            new_relations.append(rel)
    if not found:
        raise KeyError(f"关系 {rel_id} 不存在")
    return replace(
        dataset,
        relations=new_relations,
        relation_version=_bump_version(dataset.relation_version),
    )


def merge_subjects(
    dataset: Dataset,
    subject_a: str,
    subject_b: str,
    rel_type: str = "same_as",
) -> Dataset:
    """合并两个主体：新增一条 active 关系把它们连通，关系版本递增。

    页面上的"合并关系"动作落到这里——不删除旧边，只加一条可审计的新边。
    """
    known = {s.subject_id for s in dataset.samples}
    missing = [x for x in (subject_a, subject_b) if x not in known]
    if missing:
        raise KeyError(f"主体不存在于样本中: {', '.join(missing)}")
    if subject_a == subject_b:
        raise ValueError("不能合并同一个主体")

    a, b = sorted((subject_a, subject_b))
    new_id = f"rel_merge_{a}__{b}"
    if any(r.rel_id == new_id for r in dataset.relations):
        raise ValueError(f"合并关系已存在: {new_id}")
    # 已通过其他 active 边连通时仍允许添加（显式声明 same_as），
    # 连通组结果不变，但审计轨迹保留。
    new_relations = [
        *dataset.relations,
        Relation(
            rel_id=new_id,
            src_subject=a,
            dst_subject=b,
            rel_type=rel_type,
            active=True,
        ),
    ]
    return replace(
        dataset,
        relations=new_relations,
        relation_version=_bump_version(dataset.relation_version),
    )


def _bump_version(version: str) -> str:
    """relations-v2 -> relations-v3；非标准命名直接追加 -r1/-r2。"""
    if "-v" in version:
        head, _, tail = version.rpartition("-v")
        if tail.isdigit():
            return f"{head}-v{int(tail) + 1}"
    return f"{version}-r1"


# ---------------------------------------------------------------------------
# 主体 -> 集合 的索引（基于已生成版本的连通组记录）
# ---------------------------------------------------------------------------

def subject_split_map(generation: Generation) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for g in generation.groups:
        sp = g["assigned_split"]
        if sp is None:
            continue
        for subject in g["subjects"]:
            mapping[subject] = sp
    return mapping


def group_id_split_map(generation: Generation) -> dict[str, str | None]:
    return {g["group_id"]: g["assigned_split"] for g in generation.groups}


# ---------------------------------------------------------------------------
# 潜在泄漏分析
# ---------------------------------------------------------------------------

def analyze_leakage(
    old: Generation,
    old_dataset: Dataset,
    revised: Dataset,
    config: SplitConfig,
) -> dict:
    """对比旧版生成与修订后数据集，找出跨集合连通冲突。

    判定逻辑：用修订后的关系重建连通组，若新组内含有的主体在旧版中分属
    不同集合（train/val/test），则这些样本若继续使用旧清单就构成泄漏——
    训练集会经关系路径"看到"验证/测试主体。
    """
    relation_types = (
        set(config.relation_types) if config.relation_types is not None else None
    )
    new_groups, skipped = build_groups(
        revised.samples, revised.relations, relation_types
    )
    old_subject_split = subject_split_map(old)
    old_group_of_subject: dict[str, str] = {}
    for g in old.groups:
        for subject in g["subjects"]:
            old_group_of_subject[subject] = g["group_id"]

    old_rel_ids = {r.rel_id: r for r in old_dataset.relations}
    new_rel_ids = {r.rel_id: r for r in revised.relations}
    added = [
        {"rel_id": rid, "src": r.src_subject, "dst": r.dst_subject,
         "rel_type": r.rel_type, "active": r.active}
        for rid, r in sorted(new_rel_ids.items())
        if rid not in old_rel_ids
        or (old_rel_ids[rid].active != r.active and r.active)
    ]
    withdrawn = [
        {"rel_id": rid, "src": r.src_subject, "dst": r.dst_subject,
         "rel_type": r.rel_type}
        for rid, r in sorted(new_rel_ids.items())
        if rid in old_rel_ids and old_rel_ids[rid].active and not r.active
    ]

    merged_components: list[dict] = []
    conflicts: list[dict] = []
    affected_samples: set[str] = set()
    samples_by_subject: dict[str, list[str]] = {}
    for s in revised.samples:
        samples_by_subject.setdefault(s.subject_id, []).append(s.sample_id)

    for ng in new_groups:
        old_group_ids = sorted(
            {old_group_of_subject[sub] for sub in ng.subjects
             if sub in old_group_of_subject}
        )
        if len(old_group_ids) <= 1:
            continue  # 组结构未发生合并
        old_splits = sorted(
            {
                old_subject_split[sub]
                for sub in ng.subjects
                if sub in old_subject_split
            }
        )
        component = {
            "new_group_id": ng.group_id,
            "subjects": list(ng.subjects),
            "old_group_ids": old_group_ids,
            "old_splits": old_splits,
            "edge_ids": list(ng.edge_ids),
            "leakage": len(old_splits) > 1,
        }
        merged_components.append(component)
        if len(old_splits) > 1:
            subject_leak = [
                sub for sub in ng.subjects if sub in old_subject_split
            ]
            for sub in subject_leak:
                affected_samples.update(samples_by_subject.get(sub, []))
            conflicts.append(
                {
                    "new_group_id": ng.group_id,
                    "subjects": subject_leak,
                    "old_splits": old_splits,
                    "old_group_ids": old_group_ids,
                    "connecting_edges": list(ng.edge_ids),
                    "sample_count": len(
                        [
                            sid
                            for sub in subject_leak
                            for sid in samples_by_subject.get(sub, [])
                        ]
                    ),
                    "message": (
                        f"新关系使 {len(subject_leak)} 个主体连通，而它们在旧清单中"
                        f"分属 { '/'.join(old_splits) }；旧清单继续使用将导致"
                        f"训练侧经关系路径接触验证/测试主体"
                    ),
                }
            )

    return {
        "old_generation_id": old.generation_id,
        "old_freeze_id": old.freeze_id,
        "old_relation_version": old.relation_version,
        "new_relation_version": revised.relation_version,
        "added_relations": added,
        "withdrawn_relations": withdrawn,
        "skipped_relations": skipped,
        "merged_components": merged_components,
        "cross_split_conflicts": conflicts,
        "affected_sample_ids": sorted(affected_samples),
        "affected_sample_count": len(affected_samples),
        "leakage": bool(conflicts),
        "summary": (
            f"发现 {len(conflicts)} 处跨集合连通冲突，涉及 "
            f"{len(affected_samples)} 个样本；旧清单已保留未改动，"
            f"请审阅新候选版本后发布"
            if conflicts
            else "关系发生变化但未造成跨集合连通，旧清单无泄漏风险"
        ),
    }


def _generation_relations_placeholder() -> None:
    """关系 diff 以 old_dataset/revised 两个 Dataset 为准（见 analyze_leakage）。"""


def build_candidate(
    old: Generation,
    old_dataset: Dataset,
    revised: Dataset,
    config: SplitConfig,
) -> tuple[Generation, dict]:
    """基于修订后数据生成新候选版，并同时给出泄漏报告。"""
    report = analyze_leakage(old, old_dataset, revised, config)
    candidate = generate_manifests(revised, config)
    return candidate, report


def compare_generations(old: Generation, new: Generation) -> dict:
    """逐样本、逐组对比两个不可变版本的分配差异。"""
    old_sample_split = {
        row["sample_id"]: sp
        for sp in SPLITS
        for row in old.manifests[sp]
    }
    new_sample_split = {
        row["sample_id"]: sp
        for sp in SPLITS
        for row in new.manifests[sp]
    }
    moved: list[dict] = []
    for sid in sorted(set(old_sample_split) | set(new_sample_split)):
        before = old_sample_split.get(sid, "quarantine")
        after = new_sample_split.get(sid, "quarantine")
        if before != after:
            moved.append({"sample_id": sid, "from": before, "to": after})

    old_group_split = group_id_split_map(old)
    new_group_split = group_id_split_map(new)
    group_moves = [
        {"group_id": gid, "from": old_group_split.get(gid),
         "to": new_group_split.get(gid)}
        for gid in sorted(set(old_group_split) | set(new_group_split))
        if old_group_split.get(gid) != new_group_split.get(gid)
    ]

    return {
        "old_generation_id": old.generation_id,
        "new_generation_id": new.generation_id,
        "old_freeze_id": old.freeze_id,
        "new_freeze_id": new.freeze_id,
        "old_relation_version": old.relation_version,
        "new_relation_version": new.relation_version,
        "samples_moved": moved,
        "samples_moved_count": len(moved),
        "groups_moved": group_moves,
        "ratio_table": {
            "target": new.allocation["target_ratios"],
            "old_actual": old.allocation["actual_ratios"],
            "new_actual": new.allocation["actual_ratios"],
        },
        "sample_count_table": {
            "old": old.allocation["split_samples"],
            "new": new.allocation["split_samples"],
        },
    }
