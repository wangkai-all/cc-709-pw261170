"""连通组构建测试：直接关系、间接多跳、类型过滤、撤回边与脏数据跳过。"""
import unittest

from splitter.dataset import Dataset, Relation, Sample
from splitter.graph import build_groups, make_group_id


def s(sid, subject, day="2025-01-01"):
    return Sample(sid, subject, day, {})


def r(rid, a, b, typ="family", active=True):
    return Relation(rid, a, b, typ, active)


class GraphTest(unittest.TestCase):
    def test_direct_and_three_hop_indirect(self):
        """C01 与 C04 没有直接边，经三跳间接连通，必须同组。"""
        samples = [s(f"s{i}", sub) for i, sub in enumerate(
            ["C01", "C02", "C03", "C04", "X1"], start=1)]
        relations = [
            r("e1", "C01", "C02", "family"),
            r("e2", "C02", "C03", "household"),
            r("e3", "C03", "C04", "same_address"),
        ]
        groups, skipped = build_groups(samples, relations)
        self.assertEqual(skipped, [])
        by_subject = {sub: g.group_id for g in groups for sub in g.subjects}
        self.assertEqual(by_subject["C01"], by_subject["C04"])
        self.assertNotEqual(by_subject["C01"], by_subject["X1"])
        chain = next(g for g in groups if "C01" in g.subjects)
        self.assertEqual(chain.subject_count, 4)
        self.assertEqual(set(chain.edge_ids), {"e1", "e2", "e3"})

    def test_undirected_edge_direction(self):
        """关系按无向处理：反向声明同样连通。"""
        samples = [s("s1", "A"), s("s2", "B")]
        groups, _ = build_groups(samples, [r("e1", "B", "A")])
        self.assertEqual(len(groups), 1)

    def test_relation_type_filter(self):
        samples = [s("s1", "A"), s("s2", "B"), s("s3", "C")]
        relations = [r("e1", "A", "B", "family"), r("e2", "B", "C", "household")]
        groups, _ = build_groups(samples, relations, {"family"})
        by_subject = {sub: g.group_id for g in groups for sub in g.subjects}
        self.assertEqual(by_subject["A"], by_subject["B"])
        self.assertNotEqual(by_subject["A"], by_subject["C"])

    def test_inactive_relation_does_not_connect(self):
        samples = [s("s1", "A"), s("s2", "B")]
        groups, _ = build_groups(samples, [r("e1", "A", "B", active=False)])
        self.assertEqual(len(groups), 2)

    def test_dangling_relation_skipped(self):
        samples = [s("s1", "A")]
        groups, skipped = build_groups(samples, [r("e1", "A", "GHOST")])
        self.assertEqual(skipped, ["e1"])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].subjects, ("A",))

    def test_group_id_deterministic_and_changes_with_membership(self):
        gid_a = make_group_id(("A", "B"))
        gid_b = make_group_id(("A", "B"))
        gid_c = make_group_id(("A", "C"))
        self.assertEqual(gid_a, gid_b)
        self.assertNotEqual(gid_a, gid_c)

    def test_isolated_subjects_become_singleton_groups(self):
        samples = [s("s1", "A"), s("s2", "B"), s("s3", "C")]
        groups, _ = build_groups(samples, [])
        self.assertEqual(len(groups), 3)
        self.assertTrue(all(g.subject_count == 1 for g in groups))


if __name__ == "__main__":
    unittest.main()
