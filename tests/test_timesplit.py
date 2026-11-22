"""时间约束测试：截止日、shift 后移、isolate 隔离、绝不拆组。"""
import unittest

from splitter.graph import build_groups
from splitter.timesplit import CROSS_RULES, SPLITS, TimePolicy, evaluate_groups
from tests.test_graph import r, s


class TimeSplitTest(unittest.TestCase):
    def _groups(self, samples, relations=None):
        return build_groups(samples, relations or [])[0]

    def test_future_sample_never_eligible_for_train(self):
        samples = [
            s("s1", "A", "2025-06-30"),  # 截止日当天，允许
            s("s2", "B", "2025-07-01"),  # 晚一天，不允许进 train
        ]
        groups = self._groups(samples)
        policy = TimePolicy(train_cutoff="2025-06-30", val_cutoff="2025-09-30")
        ev = evaluate_groups(groups, samples, policy)
        gid_a = next(g.group_id for g in groups if "A" in g.subjects)
        gid_b = next(g.group_id for g in groups if "B" in g.subjects)
        self.assertTrue(ev.infos[gid_a].eligible["train"])
        self.assertFalse(ev.infos[gid_b].eligible["train"])
        self.assertTrue(ev.infos[gid_b].eligible["test"])

    def test_cross_boundary_group_shift_never_split(self):
        """同组样本跨越 train 边界：整组强制后移，组不出现在更早集合。"""
        samples = [s("s1", "A", "2025-05-01"), s("s2", "A", "2025-08-01")]
        groups = self._groups(samples)
        policy = TimePolicy("2025-06-30", "2025-09-30", "shift")
        ev = evaluate_groups(groups, samples, policy)
        gid = groups[0].group_id
        self.assertTrue(ev.infos[gid].crosses_boundary)
        self.assertEqual(ev.forced_split[gid], "val")
        self.assertNotIn(gid, ev.quarantined)

    def test_cross_boundary_group_isolate(self):
        samples = [s("s1", "A", "2025-05-01"), s("s2", "A", "2025-10-01")]
        groups = self._groups(samples)
        policy = TimePolicy("2025-06-30", "2025-09-30", "isolate")
        ev = evaluate_groups(groups, samples, policy)
        gid = groups[0].group_id
        self.assertIn(gid, ev.quarantined)
        self.assertNotIn(gid, ev.forced_split)

    def test_eligibility_is_nested_and_train_eligible_groups_stay_free(self):
        """资格必然是后缀链：train 合格 ⇒ val 合格 ⇒ test 合格；
        train 合格的组不被时间强制，交给比例分配。"""
        samples = [
            s("s1", "A", "2025-05-01"), s("s2", "A", "2025-06-30"),  # 全在 train 窗
            s("s3", "B", "2025-05-01"), s("s4", "B", "2025-10-01"),  # 越过两个边界
        ]
        groups = self._groups(samples)
        policy = TimePolicy("2025-06-30", "2025-09-30", "shift")
        ev = evaluate_groups(groups, samples, policy)
        for info in ev.infos.values():
            if info.eligible["train"]:
                self.assertTrue(info.eligible["val"])
                self.assertTrue(info.eligible["test"])
            if info.eligible["val"]:
                self.assertTrue(info.eligible["test"])
        gid_a = next(g.group_id for g in groups if "A" in g.subjects)
        gid_b = next(g.group_id for g in groups if "B" in g.subjects)
        self.assertNotIn(gid_a, ev.forced_split)  # train 合格 -> 自由组
        self.assertEqual(ev.forced_split[gid_b], "test")  # 越过两界 -> 强制 test

    def test_no_cutoff_means_no_time_restriction(self):
        samples = [s("s1", "A", "2030-01-01")]
        groups = self._groups(samples)
        ev = evaluate_groups(groups, samples, TimePolicy())
        self.assertTrue(ev.missing_cutoff)
        for g in groups:
            for sp in SPLITS:
                self.assertTrue(ev.infos[g.group_id].eligible[sp])

    def test_invalid_rule_rejected(self):
        with self.assertRaises(ValueError):
            TimePolicy(cross_boundary_rule="explode")
        self.assertEqual(CROSS_RULES, ("shift", "isolate"))

    def test_val_cutoff_before_train_rejected(self):
        with self.assertRaises(ValueError):
            TimePolicy(train_cutoff="2025-09-30", val_cutoff="2025-06-30")


if __name__ == "__main__":
    unittest.main()
