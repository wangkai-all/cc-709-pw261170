"""CSV 解析与数据校验测试。"""
import unittest

from splitter.dataset import (
    DatasetError,
    content_hash,
    dump_relations_csv,
    dump_samples_csv,
    load_relations_csv,
    load_samples_csv,
)

SAMPLES_CSV = (
    "sample_id,subject_id,event_date,amount\n"
    "s1,A,2025-01-01,100\n"
    "s2,A,2025-02-01,200\n"
    "s3,B,2025-03-01,300\n"
)
RELATIONS_CSV = (
    "rel_id,src_subject,dst_subject,rel_type,active\n"
    "r1,A,B,family,true\n"
    "r2,B,C,household,false\n"
)


class DatasetCsvTest(unittest.TestCase):
    def test_roundtrip_samples(self):
        samples = load_samples_csv(SAMPLES_CSV)
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].attributes, {"amount": "100"})
        # 确定性输出：排序、统一换行
        out = dump_samples_csv(samples)
        self.assertIn("amount", out.splitlines()[0])
        self.assertEqual(content_hash(out), content_hash(out.replace("\n", "\r\n")))

    def test_roundtrip_relations(self):
        rels = load_relations_csv(RELATIONS_CSV)
        self.assertTrue(rels[0].active)
        self.assertFalse(rels[1].active)
        out = dump_relations_csv(rels)
        self.assertIn("false", out)
        self.assertEqual(load_relations_csv(out), rels)

    def test_missing_column(self):
        with self.assertRaises(DatasetError):
            load_samples_csv("sample_id,subject_id\ns1,A\n")

    def test_duplicate_sample_id(self):
        with self.assertRaises(DatasetError):
            load_samples_csv(
                "sample_id,subject_id,event_date\ns1,A,2025-01-01\ns1,B,2025-01-02\n"
            )

    def test_bad_date(self):
        with self.assertRaises(DatasetError):
            load_samples_csv("sample_id,subject_id,event_date\ns1,A,01/01/2025\n")

    def test_active_defaults_true(self):
        rels = load_relations_csv(
            "rel_id,src_subject,dst_subject,rel_type\nr1,A,B,family\n"
        )
        self.assertTrue(rels[0].active)


if __name__ == "__main__":
    unittest.main()
