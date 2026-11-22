"""冻结与清单测试：freeze_id 指纹、逐字节可复现、组不拆分、隔离清单。"""
import unittest

from splitter.examples import build_example_dataset, example_config
from splitter.manifest import SplitConfig, compute_freeze_id, generate_manifests
from splitter.timesplit import SPLITS


class ManifestTest(unittest.TestCase):
    def setUp(self):
        self.dataset = build_example_dataset()
        self.config = SplitConfig(**example_config())

    def test_reproducible_byte_identical(self):
        first = generate_manifests(self.dataset, self.config)
        second = generate_manifests(self.dataset, self.config)
        self.assertEqual(first.freeze_id, second.freeze_id)
        self.assertEqual(first.generation_id, second.generation_id)
        for sp in SPLITS:
            self.assertEqual(first.manifest_csv[sp], second.manifest_csv[sp])
        self.assertEqual(first.quarantine_csv, second.quarantine_csv)

    def test_freeze_id_changes_with_any_frozen_input(self):
        base = generate_manifests(self.dataset, self.config)
        # 改种子
        cfg_seed = SplitConfig(**{**example_config(), "seed": self.config.seed + 1})
        self.assertNotEqual(base.freeze_id, generate_manifests(self.dataset, cfg_seed).freeze_id)
        # 改比例
        cfg_ratio = SplitConfig(**{**example_config(), "target_ratios": {"train": 0.5, "val": 0.25, "test": 0.25}})
        self.assertNotEqual(base.freeze_id, generate_manifests(self.dataset, cfg_ratio).freeze_id)
        # 改时间边界
        cfg_cut = SplitConfig(**{**example_config(), "train_cutoff": "2025-03-31"})
        self.assertNotEqual(base.freeze_id, generate_manifests(self.dataset, cfg_cut).freeze_id)
        # 改跨边界规则
        cfg_rule = SplitConfig(**{**example_config(), "cross_boundary_rule": "isolate"})
        self.assertNotEqual(base.freeze_id, generate_manifests(self.dataset, cfg_rule).freeze_id)

    def test_freeze_id_uses_content_hashes_not_filenames(self):
        fid1 = compute_freeze_id(self.dataset, self.config, "h1", "h2")
        fid2 = compute_freeze_id(self.dataset, self.config, "h1", "h2")
        fid3 = compute_freeze_id(self.dataset, self.config, "h1", "different")
        self.assertEqual(fid1, fid2)
        self.assertNotEqual(fid1, fid3)

    def test_no_group_split_inside_manifests(self):
        """同组的所有样本必须出现在同一份清单里（核心不变量）。"""
        gen = generate_manifests(self.dataset, self.config)
        group_splits = {}
        for sp in SPLITS:
            for row in gen.manifests[sp]:
                gid = row["group_id"]
                if gid in group_splits and group_splits[gid] != sp:
                    self.fail(
                        f"组 {gid} 被拆分到 {group_splits[gid]} 和 {sp} 两个集合"
                    )
                group_splits[gid] = sp
        # 示例中每个非隔离组都有去向
        non_quarantine = [g for g in gen.groups if not g["quarantined"]]
        self.assertEqual(len(group_splits), len(non_quarantine))

    def test_subjects_never_cross_splits(self):
        """同一主体的所有样本必须在同一集合。"""
        gen = generate_manifests(self.dataset, self.config)
        subject_splits = {}
        for sp in SPLITS:
            for row in gen.manifests[sp]:
                seen = subject_splits.setdefault(row["subject_id"], sp)
                self.assertEqual(seen, sp)

    def test_manifests_cover_all_non_quarantined_samples(self):
        gen = generate_manifests(self.dataset, self.config)
        listed = sum(len(gen.manifests[sp]) for sp in SPLITS)
        self.assertEqual(listed + len(gen.quarantine), len(self.dataset.samples))

    def test_isolate_rule_produces_quarantine_csv(self):
        cfg = SplitConfig(**{**example_config(), "cross_boundary_rule": "isolate"})
        gen = generate_manifests(self.dataset, cfg)
        self.assertGreater(len(gen.quarantine), 0)
        self.assertIn("quarantine_reason", gen.quarantine_csv)
        # 隔离样本绝不进入三份主清单
        quarantined_ids = {row["sample_id"] for row in gen.quarantine}
        listed_ids = {row["sample_id"] for sp in SPLITS for row in gen.manifests[sp]}
        self.assertTrue(quarantined_ids.isdisjoint(listed_ids))

    def test_manifest_rows_carry_attributes(self):
        gen = generate_manifests(self.dataset, self.config)
        row = gen.manifests["test"][0]
        self.assertIn("amount", row)
        self.assertIn("channel", row)
        self.assertIn("group_id", row)

    def test_example_contains_three_hop_chain_in_one_group(self):
        gen = generate_manifests(self.dataset, self.config)
        chain = next(
            g for g in gen.groups
            if {"C01", "C02", "C03", "C04"}.issubset(set(g["subjects"]))
        )
        self.assertEqual(chain["subject_count"], 4)
        # 三层间接关系的边都记录在组上
        self.assertEqual(
            set(chain["edge_ids"]),
            {"rel_c01_c02", "rel_c02_c03", "rel_c03_c04"},
        )


if __name__ == "__main__":
    unittest.main()
