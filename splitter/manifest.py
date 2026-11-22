"""冻结配置与清单生成。

一次"生成"把以下内容全部固化，freeze_id 是其 SHA-256 指纹：

- 种子 seed
- 样本版本 / 关系版本及两份输入的内容哈希
- 目标比例、时间边界、跨边界规则、参与连边的关系类型
- 穷举上限与分配策略

只要其中任何一项变化，freeze_id 就变；全部不变则重新生成逐字节一致，
从而保证"种子、样本版本、关系版和分配规则冻结后，重新生成结果相同"。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .allocate import AllocationResult, allocate, normalize_ratios
from .dataset import (
    Dataset,
    content_hash,
    dump_relations_csv,
    dump_samples_csv,
)
from .graph import Group, groups_from_dataset
from .timesplit import SPLITS, TimePolicy, evaluate_groups


@dataclass
class SplitConfig:
    seed: int = 42
    target_ratios: dict[str, float] = field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15}
    )
    train_cutoff: str | None = None
    val_cutoff: str | None = None
    cross_boundary_rule: str = "shift"
    relation_types: list[str] | None = None  # None = 全部类型
    exhaustive_limit: int = 10


def canonical_json(obj: object) -> str:
    """排序键、无多余空白的确定性 JSON。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def compute_freeze_id(
    dataset: Dataset,
    config: SplitConfig,
    sample_hash: str,
    relation_hash: str,
) -> str:
    payload = {
        "seed": config.seed,
        "sample_version": dataset.sample_version,
        "relation_version": dataset.relation_version,
        "sample_hash": sample_hash,
        "relation_hash": relation_hash,
        "target_ratios": normalize_ratios(config.target_ratios),
        "train_cutoff": config.train_cutoff,
        "val_cutoff": config.val_cutoff,
        "cross_boundary_rule": config.cross_boundary_rule,
        "relation_types": (
            sorted(config.relation_types)
            if config.relation_types is not None
            else None
        ),
        "exhaustive_limit": config.exhaustive_limit,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _manifest_rows(
    dataset: Dataset,
    groups: list[Group],
    assignment: dict[str, str],
) -> dict[str, list[dict[str, str]]]:
    group_of_subject: dict[str, Group] = {}
    for g in groups:
        for subject in g.subjects:
            group_of_subject[subject] = g

    rows: dict[str, list[dict[str, str]]] = {sp: [] for sp in SPLITS}
    for sample in sorted(dataset.samples, key=lambda s: s.sample_id):
        g = group_of_subject[sample.subject_id]
        sp = assignment.get(g.group_id)
        if sp is None:
            continue  # 隔离样本进 quarantine，不进三份清单
        rows[sp].append(
            {
                "sample_id": sample.sample_id,
                "subject_id": sample.subject_id,
                "event_date": sample.event_date,
                "group_id": g.group_id,
                **sample.attributes,
            }
        )
    return rows


def _quarantine_rows(
    dataset: Dataset,
    groups: list[Group],
    quarantined: dict[str, str],
) -> list[dict[str, str]]:
    group_of_subject = {
        subject: g for g in groups for subject in g.subjects
    }
    rows: list[dict[str, str]] = []
    for sample in sorted(dataset.samples, key=lambda s: s.sample_id):
        g = group_of_subject[sample.subject_id]
        if g.group_id in quarantined:
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "subject_id": sample.subject_id,
                    "event_date": sample.event_date,
                    "group_id": g.group_id,
                    "quarantine_reason": quarantined[g.group_id],
                    **sample.attributes,
                }
            )
    return rows


@dataclass
class Generation:
    """一次完整生成的全部产物（可直接序列化存储）。"""

    freeze_id: str
    generation_id: str
    created_at: str
    dataset_name: str
    sample_version: str
    relation_version: str
    config: dict
    sample_hash: str
    relation_hash: str
    groups: list[dict]
    skipped_relations: list[str]
    allocation: dict
    manifests: dict[str, list[dict[str, str]]]
    quarantine: list[dict[str, str]]
    manifest_csv: dict[str, str]
    quarantine_csv: str
    published: bool = False
    published_as: str | None = None


def generate_manifests(
    dataset: Dataset,
    config: SplitConfig,
    sample_hash: str | None = None,
    relation_hash: str | None = None,
) -> Generation:
    """端到端：连通组 → 时间评价 → 组级分配 → 冻结清单。"""
    sample_text = dump_samples_csv(dataset.samples)
    relation_text = dump_relations_csv(dataset.relations)
    sample_hash = sample_hash or content_hash(sample_text)
    relation_hash = relation_hash or content_hash(relation_text)

    relation_types = (
        set(config.relation_types) if config.relation_types is not None else None
    )
    groups, skipped = groups_from_dataset(dataset, relation_types)

    policy = TimePolicy(
        train_cutoff=config.train_cutoff,
        val_cutoff=config.val_cutoff,
        cross_boundary_rule=config.cross_boundary_rule,
    )
    samples_per_group: dict[str, int] = {}
    for g in groups:
        members = set(g.subjects)
        samples_per_group[g.group_id] = sum(
            1 for s in dataset.samples if s.subject_id in members
        )

    time_eval = evaluate_groups(groups, dataset.samples, policy)
    result: AllocationResult = allocate(
        groups=groups,
        samples_per_group=samples_per_group,
        time_eval=time_eval,
        target_ratios=config.target_ratios,
        seed=config.seed,
        exhaustive_limit=config.exhaustive_limit,
    )

    manifests = _manifest_rows(dataset, groups, result.assignment)
    quarantine = _quarantine_rows(dataset, groups, time_eval.quarantined)

    group_records = []
    for g in groups:
        info = time_eval.infos[g.group_id]
        group_records.append(
            {
                "group_id": g.group_id,
                "subjects": list(g.subjects),
                "edge_ids": list(g.edge_ids),
                "subject_count": g.subject_count,
                "sample_count": samples_per_group[g.group_id],
                "min_date": info.min_date,
                "max_date": info.max_date,
                "eligible_splits": [sp for sp in SPLITS if info.eligible[sp]],
                "crosses_boundary": info.crosses_boundary,
                "limiting_cutoff": info.limiting_cutoff,
                "assigned_split": result.assignment.get(g.group_id),
                "quarantined": g.group_id in time_eval.quarantined,
                "reason": result.reasons.get(g.group_id, ""),
            }
        )
    group_records.sort(key=lambda r: (-r["sample_count"], r["group_id"]))

    import csv
    import io

    def rows_to_csv(rows: list[dict[str, str]]) -> str:
        if not rows:
            return ""
        cols: list[str] = []
        for row in rows:
            for k in row:
                if k not in cols:
                    cols.append(k)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return buf.getvalue()

    manifest_csv = {sp: rows_to_csv(manifests[sp]) for sp in SPLITS}
    quarantine_csv = rows_to_csv(quarantine)

    freeze_id = compute_freeze_id(dataset, config, sample_hash, relation_hash)
    return Generation(
        freeze_id=freeze_id,
        generation_id=f"gen_{freeze_id[:12]}",
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dataset_name=dataset.name,
        sample_version=dataset.sample_version,
        relation_version=dataset.relation_version,
        config=asdict(config),
        sample_hash=sample_hash,
        relation_hash=relation_hash,
        groups=group_records,
        skipped_relations=skipped,
        allocation={
            "strategy": result.strategy,
            "exhaustive_limit": result.exhaustive_limit,
            "greedy_sse": result.greedy_sse,
            "sse": result.sse,
            "target_ratios": result.target_ratios,
            "actual_ratios": result.actual_ratios,
            "target_samples": result.target_samples,
            "split_samples": result.split_samples,
            "split_subjects": result.split_subjects,
            "deviation": result.deviation,
            "notes": result.notes,
            "reasons": result.reasons,
        },
        manifests=manifests,
        quarantine=quarantine,
        manifest_csv=manifest_csv,
        quarantine_csv=quarantine_csv,
    )
