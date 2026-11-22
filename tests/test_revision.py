"""关系修订与泄漏分析测试：旧清单不可变、迟到修订检出冲突、版本对比。"""
import unittest

from splitter.examples import LATE_REVISION, build_example_dataset, example_config
from splitter.manifest import SplitConfig, generate_manifests
from splitter.revision import (
    analyze_leakage,
    build_candidate,
    compare_generations,
    merge_subjects,
    subject_split_map,
    withdraw_relation,
)
from splitter.timesplit import SPLITS


class RevisionTest(unittest.TestCase):
    def setUp(self):
        self.dataset = build_example_dataset()
        self.config = SplitConfig(**example_config())
        self.old = generate_manifests(self.dataset, self.config)

    def test_late_revision_detects_cross_split_leakage(self):
        revised = merge_subjects(
            self.dataset,
            LATE_REVISION["subject_a"],
            LATE_REVISION["subject_b"],
            LATE_REVISION["rel_type"],
        )
        self.assertEqual(revised.relation_version, "relations-v2")
        report = analyze_leakage(self.old, self.dataset, revised, self.config)
        self.assertTrue(report["leakage"])
        self.assertEqual(len(report["cross_split_conflicts"]), 1)
        conflict = report["cross_split_conflicts"][0]
        self.assertEqual(set(conflict["old_splits"]), {"train", "test"})
        self.assertIn("rel_merge_C04__S01", conflict["connecting_edges"])
        # 受影响样本至少包含 S01 与 chain 组的全部样本
        self.assertGreaterEqual(conflict["sample_count"], 12)

    def test_old_manifests_remain_untouched_after_revision(self):
        """核心承诺：修订只能产出新候选，旧 Generation 内容保持原样。"""
        old_csv = {sp: self.old.manifest_csv[sp] for sp in SPLITS}
        old_freeze = self.old.freeze_id
        revised = merge_subjects(self.dataset, "C04", "S01", "same_as")
        candidate, report = build_candidate(self.old, self.dataset, revised, self.config)
        self.assertTrue(report["leakage"])
        self.assertNotEqual(candidate.freeze_id, old_freeze)
        for sp in SPLITS:
            self.assertEqual(self.old.manifest_csv[sp], old_csv[sp])
        self.assertEqual(self.old.freeze_id, old_freeze)
        self.assertFalse(self.old.published)

    def test_withdraw_relation_is_auditable(self):
        revised = withdraw_relation(self.dataset, "rel_c03_c04")
        self.assertEqual(revised.relation_version, "relations-v2")
        edge = next(r for r in revised.relations if r.rel_id == "rel_c03_c04")
        self.assertFalse(edge.active)
        # 撤回后 C04 不再与 C01 同组
        report = analyze_leakage(self.old, self.dataset, revised, self.config)
        merged = [c for c in report["merged_components"]
                  if {"C01", "C04"}.issubset(set(c["subjects"]))]
        self.assertEqual(merged, [])
        self.assertEqual(len(report["withdrawn_relations"]), 1)
        # 撤回不产生跨集合冲突（边只会减少连通）
        self.assertFalse(report["leakage"])

    def test_double_withdraw_rejected(self):
        revised = withdraw_relation(self.dataset, "rel_c03_c04")
        with self.assertRaises(ValueError):
            withdraw_relation(revised, "rel_c03_c04")

    def test_merge_unknown_subject_rejected(self):
        with self.assertRaises(KeyError):
            merge_subjects(self.dataset, "C04", "NOPE")

    def test_compare_generations_lists_moved_samples(self):
        revised = merge_subjects(self.dataset, "C04", "S01", "same_as")
        candidate, _ = build_candidate(self.old, self.dataset, revised, self.config)
        cmp = compare_generations(self.old, candidate)
        self.assertGreater(cmp["samples_moved_count"], 0)
        moved_ids = {m["sample_id"] for m in cmp["samples_moved"]}
        # S01 的三个样本从 train 移走
        self.assertTrue(any(mid.startswith("s_S01") for mid in moved_ids))
        for move in cmp["samples_moved"]:
            self.assertNotEqual(move["from"], move["to"])
        self.assertEqual(set(cmp["ratio_table"].keys()),
                         {"target", "old_actual", "new_actual"})

    def test_subject_split_map_covers_assigned_groups(self):
        mapping = subject_split_map(self.old)
        # C01..C04 同组，必须映射到同一集合
        vals = {mapping[sub] for sub in ("C01", "C02", "C03", "C04")}
        self.assertEqual(len(vals), 1)

    def test_no_revision_no_leakage(self):
        report = analyze_leakage(self.old, self.dataset, self.dataset, self.config)
        self.assertFalse(report["leakage"])
        self.assertEqual(report["cross_split_conflicts"], [])


if __name__ == "__main__":
    unittest.main()
