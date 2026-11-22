"""存储层测试：版本目录、不可变生成、发布版本只增不减。"""
import tempfile
import unittest
from pathlib import Path

from splitter.examples import build_example_dataset, example_config
from splitter.manifest import SplitConfig, generate_manifests
from splitter.revision import merge_subjects
from splitter.store import Store, read_json


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(self.tmp.name)
        self.dataset = build_example_dataset()

    def tearDown(self):
        self.tmp.cleanup()

    def _save_example(self):
        from splitter.dataset import dump_relations_csv, dump_samples_csv
        return self.store.create_dataset(
            "lake_demo",
            dump_samples_csv(self.dataset.samples),
            dump_relations_csv(self.dataset.relations),
        )

    def test_dataset_versioned_directories(self):
        self._save_example()
        base = Path(self.tmp.name) / "datasets" / "lake_demo"
        self.assertTrue((base / "versions" / "samples-v1__relations-v1" / "samples.csv").exists())
        revised = merge_subjects(self.dataset, "C04", "S01", "same_as")
        self.store.save_dataset_revision(revised)
        self.assertTrue((base / "versions" / "samples-v1__relations-v2" / "relations.csv").exists())
        # 当前指针已更新，历史版本目录仍在
        loaded = self.store.load_dataset("lake_demo")
        self.assertEqual(loaded.relation_version, "relations-v2")
        old = self.store.load_dataset("lake_demo", "samples-v1", "relations-v1")
        self.assertEqual(len(old.relations), len(self.dataset.relations))

    def test_generation_immutable_and_reloadable(self):
        self._save_example()
        config = SplitConfig(**example_config())
        gen = generate_manifests(self.dataset, config)
        self.store.save_generation(gen)
        # 同 freeze_id 重复保存不报错（幂等）
        self.store.save_generation(generate_manifests(self.dataset, config))
        loaded = self.store.load_generation(gen.generation_id)
        self.assertEqual(loaded.freeze_id, gen.freeze_id)
        self.assertEqual(loaded.manifest_csv, gen.manifest_csv)
        listing = self.store.list_generations("lake_demo")
        self.assertEqual(len(listing), 1)

    def test_publish_versions_are_append_only(self):
        self._save_example()
        config = SplitConfig(**example_config())
        gen1 = generate_manifests(self.dataset, config)
        self.store.save_generation(gen1)
        pub1 = self.store.publish(gen1.generation_id)
        self.assertEqual(pub1["published_as"], "v1")

        revised = merge_subjects(self.dataset, "C04", "S01", "same_as")
        self.store.save_dataset_revision(revised)
        gen2 = generate_manifests(revised, config)
        self.store.save_generation(gen2)
        pub2 = self.store.publish(gen2.generation_id)
        self.assertEqual(pub2["published_as"], "v2")

        pubs = self.store.list_publications("lake_demo")
        self.assertEqual([p["published_as"] for p in pubs], ["v1", "v2"])
        # v1 的发布文件没有被 v2 覆盖
        v1 = read_json(
            Path(self.tmp.name) / "publications" / "lake_demo" / "v1" / "generation.json"
        )
        self.assertEqual(v1["freeze_id"], gen1.freeze_id)
        # generation 上的发布标记已回写
        self.assertTrue(self.store.load_generation(gen1.generation_id).published)

    def test_leakage_report_persisted_with_candidate(self):
        self._save_example()
        report = {"leakage": True, "summary": "test"}
        path = self.store.save_leakage_report("gen_abc", report)
        self.assertTrue(path.exists())
        self.assertEqual(self.store.load_leakage_report("gen_abc"), report)
        self.assertIsNone(self.store.load_leakage_report("missing"))


if __name__ == "__main__":
    unittest.main()
