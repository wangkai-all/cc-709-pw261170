"""主体关系图与连通组。

关键约束：同一主体、以及通过指定类型关系**间接**连通的主体必须落在同一份
清单里。因此先在无向图上求连通组，后续所有分配都以组为最小单位，绝不拆组。

使用并查集（Union-Find）实现，组 ID 由成员列表的 SHA-256 派生，
成员或边发生变化时组 ID 必然变化，便于版本间追踪合并/拆分。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .dataset import Dataset, Relation, Sample


class _UnionFind:
    def __init__(self, nodes: list[str]):
        self.parent = {n: n for n in nodes}
        self.rank = {n: 0 for n in nodes}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # 路径压缩
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass(frozen=True)
class Group:
    """一个连通组：组内主体及连接它们的关系 ID。"""

    group_id: str
    subjects: tuple[str, ...]
    edge_ids: tuple[str, ...]

    @property
    def subject_count(self) -> int:
        return len(self.subjects)


def make_group_id(subjects: tuple[str, ...]) -> str:
    digest = hashlib.sha256("|".join(subjects).encode("utf-8")).hexdigest()
    return f"g_{digest[:10]}"


def build_groups(
    samples: list[Sample],
    relations: list[Relation],
    relation_types: set[str] | None = None,
) -> tuple[list[Group], list[str]]:
    """根据启用的关系类型构建连通组。

    - 没有任何关系的主体成为单成员组，保证每个样本都归属某个组。
    - 非 active 的关系（已撤回）不参与连边。
    - 端点不在样本主体集合内的关系被跳过并记录，避免脏数据静默扩组。

    返回 ``(按最小主体 ID 排序的组列表, 被跳过的关系 ID 列表)``。
    """
    universe = sorted({s.subject_id for s in samples})
    universe_set = set(universe)
    uf = _UnionFind(universe)

    skipped: list[str] = []
    used_edges: dict[str, set[str]] = {}
    for rel in sorted(relations, key=lambda r: r.rel_id):
        if not rel.active:
            continue
        if relation_types is not None and rel.rel_type not in relation_types:
            continue
        if rel.src_subject not in universe_set or rel.dst_subject not in universe_set:
            skipped.append(rel.rel_id)
            continue
        uf.union(rel.src_subject, rel.dst_subject)

    members: dict[str, list[str]] = {}
    edges: dict[str, list[str]] = {}
    for subject in universe:
        root = uf.find(subject)
        members.setdefault(root, []).append(subject)

    # 记录每个组内部的边
    for rel in sorted(relations, key=lambda r: r.rel_id):
        if not rel.active:
            continue
        if relation_types is not None and rel.rel_type not in relation_types:
            continue
        if rel.src_subject not in universe_set or rel.dst_subject not in universe_set:
            continue
        root = uf.find(rel.src_subject)
        edges.setdefault(root, []).append(rel.rel_id)

    groups: list[Group] = []
    for root in sorted(members, key=lambda r: members[r][0]):
        subjects = tuple(sorted(members[root]))
        groups.append(
            Group(
                group_id=make_group_id(subjects),
                subjects=subjects,
                edge_ids=tuple(sorted(edges.get(root, []))),
            )
        )
    return groups, skipped


def groups_from_dataset(
    dataset: Dataset, relation_types: set[str] | None = None
) -> tuple[list[Group], list[str]]:
    return build_groups(dataset.samples, dataset.relations, relation_types)
