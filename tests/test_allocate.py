"""组级分配测试：独立暴力穷举校验最优性、硬约束与可复现性。

这里的穷举器与生产代码 ``splitter.allocate._exhaustive`` 互相独立实现，
只读取组规模与时间资格，枚举全部组->集合映射后取 SSE 最小者，
用来交叉验证生产分配器确实找到全局最优（而非仅局部贪心）。
"""
import itertools
import unittest

from splitter.allocate import allocate, normalize_ratios
from splitter.graph import build_groups
from splitter.timesplit import SPLITS, TimePolicy, evaluate_groups
from tests.test_graph import r, s


def brute_force_best(stats, target_samples):
    """独立参考实现：枚举全部可行的组级分配，返回 (最小SSE, 最优分配集合)。"""
    domains = [list(st.eligible_splits) for st in stats]
    best_sse = float("inf")
    best_assignments = []
    for combo in itertools.product(*domains):
        counts = {sp: 0 for sp in SPLITS}
        assignment = {}
        for st, sp in zip(stats, combo):
            assignment[st.group.group_id] = sp
            counts[sp] += st.sample_count
        sse = sum((counts[sp] - target_samples[sp]) ** 2 for sp in SPLITS)
        if sse < best_sse - 1e-9:
            best_sse = sse
            best_assignments = [frozenset(assignment.items())]
        elif abs(sse - best_sse) <= 1e-9:
            best_assignments.append(frozenset(assignment.items()))
    return best_sse, best_assignments


def build_eval(samples, relations, train_cutoff="2025-06-30", val_cutoff="2025-09-30", rule="shift"):
    groups, skipped = build_groups(samples, relations)
    assert not skipped
    policy = TimePolicy(train_cutoff, val_cutoff, rule)
    ev = evaluate_groups(groups, samples, policy)
    spg = {}
    for g in groups:
        members = set(g.subjects)
        spg[g.group_id] = sum(1 for x in samples if x.subject_id in members)
    return groups, spg, ev


class AllocateExhaustiveTest(unittest.TestCase):
    def setUp(self):
        # 6 个独立主体 + 1 个二主体连通组，共 7 个组，全部落在 train 窗内 -> 全自由
        samples = [s(f"s{i}", f"A{i}", f"2025-0{((i % 6) + 1)}-10") for i in range(6)]
        samples += [s("s6", "B1", "2025-03-01"), s("s7", "B2", "2025-03-02")]
        relations = [r("eB", "B1", "B2", "household")]
        self.samples, self.relations = samples, relations
        self.ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

    def test_matches_independent_brute_force(self):
        groups, spg, ev = build_eval(self.samples, self.relations)
        result = allocate(groups, spg, ev, self.ratios, seed=7)
        self.assertEqual(result.strategy, "exhaustive")

        # 独立穷举（忽略强制层，此处无强制组；资格域全部为三集合）
        from splitter.allocate import _build_stats
        stats = _build_stats(groups, spg, ev)
        allocatable = sum(st.sample_count for st in stats)
        target = {sp: normalize_ratios(self.ratios)[sp] * allocatable for sp in SPLITS}
        best_sse, best_set = brute_force_best(stats, target)
        self.assertAlmostEqual(result.sse, best_sse, places=6)
        self.assertIn(frozenset(result.assignment.items()), best_set)

    def test_hard_constraints_every_group_whole_and_eligible(self):
        groups, spg, ev = build_eval(self.samples, self.relations)
        result = allocate(groups, spg, ev, self.ratios, seed=11)
        for g in groups:
            self.assertIn(g.group_id, result.assignment, "每个非隔离组必须有去向")
            sp = result.assignment[g.group_id]
            self.assertIn(sp, SPLITS)
            self.assertTrue(
                ev.infos[g.group_id].eligible[sp],
                f"组 {g.group_id} 被放进时间不合格的集合 {sp}",
            )
        # 样本总数守恒
        self.assertEqual(
            sum(result.split_samples.values()),
            sum(spg.values()),
        )
        # 主体计数守恒
        total_subjects = sum(g.subject_count for g in groups)
        self.assertEqual(sum(result.split_subjects.values()), total_subjects)

    def test_reproducible_same_seed(self):
        groups, spg, ev = build_eval(self.samples, self.relations)
        r1 = allocate(groups, spg, ev, self.ratios, seed=99)
        r2 = allocate(groups, spg, ev, self.ratios, seed=99)
        self.assertEqual(r1.assignment, r2.assignment)
        self.assertEqual(r1.sse, r2.sse)

    def test_different_seed_still_valid_and_optimal(self):
        groups, spg, ev = build_eval(self.samples, self.relations)
        assignments = set()
        for seed in (1, 2, 3, 4, 5):
            result = allocate(groups, spg, ev, self.ratios, seed=seed)
            # 每次都必须是穷举最优，种子只影响平局
            from splitter.allocate import _build_stats
            stats = _build_stats(groups, spg, ev)
            allocatable = sum(st.sample_count for st in stats)
            target = {sp: normalize_ratios(self.ratios)[sp] * allocatable for sp in SPLITS}
            best_sse, best_set = brute_force_best(stats, target)
            self.assertAlmostEqual(result.sse, best_sse, places=6)
            self.assertIn(frozenset(result.assignment.items()), best_set)
            assignments.add(tuple(sorted(result.assignment.items())))
        # 五个种子至少出现过一种分配；平局存在时种子会产生不同但等价的方案
        self.assertGreaterEqual(len(assignments), 1)

    def test_time_forced_groups_are_not_moved(self):
        """含截止日后样本的组必须整体后移，分配器不能为凑比例挪回 train。"""
        samples = [
            s("s1", "A", "2025-03-01"), s("s2", "A", "2025-11-01"),  # 跨 train 边界
            s("s3", "B", "2025-02-01"),
        ]
        groups, spg, ev = build_eval(samples, [])
        result = allocate(groups, spg, ev, self.ratios, seed=3)
        gid_a = next(g.group_id for g in groups if "A" in g.subjects)
        self.assertNotEqual(result.assignment[gid_a], "train")
        self.assertEqual(ev.forced_split[gid_a], result.assignment[gid_a])

    def test_isolated_groups_excluded_from_all_splits(self):
        samples = [
            s("s1", "A", "2025-03-01"), s("s2", "A", "2025-11-01"),
            s("s3", "B", "2025-02-01"),
        ]
        groups, spg, ev = build_eval(samples, [], rule="isolate")
        result = allocate(groups, spg, ev, self.ratios, seed=3)
        gid_a = next(g.group_id for g in groups if "A" in g.subjects)
        self.assertNotIn(gid_a, result.assignment)
        self.assertEqual(sum(result.split_samples.values()), 1)  # 只剩 B 的样本

    def test_large_group_dominates_deviation(self):
        """大组（占比 >=25%）必须在解释 notes 中被指出为偏差来源。"""
        # 一个 6 样本大组 + 4 个单样本组
        samples = [s(f"big{i}", "BIG", f"2025-01-0{(i % 9) + 1}") for i in range(6)]
        samples += [s(f"small{i}", f"S{i}", "2025-02-01") for i in range(4)]
        groups, spg, ev = build_eval(samples, [])
        result = allocate(groups, spg, ev, self.ratios, seed=1)
        self.assertTrue(any("最大连通组" in n for n in result.notes))
        # 大组整体落在一个集合
        gid_big = next(g.group_id for g in groups if "BIG" in g.subjects)
        self.assertEqual(spg[gid_big], 6)
        self.assertIn(result.assignment[gid_big], SPLITS)


class GreedyPathTest(unittest.TestCase):
    def test_greedy_used_above_limit_and_respects_constraints(self):
        # 12 个自由组超过默认穷举上限 10
        samples = [s(f"s{i}", f"A{i:02d}", "2025-03-01") for i in range(12)]
        groups, spg, ev = build_eval(samples, [])
        result = allocate(groups, spg, ev,
                          {"train": 0.7, "val": 0.15, "test": 0.15}, seed=5)
        self.assertEqual(result.strategy, "greedy")
        self.assertEqual(len(result.assignment), 12)
        self.assertEqual(sum(result.split_samples.values()), 12)
        for g in groups:
            self.assertIn(result.assignment[g.group_id], SPLITS)

    def test_greedy_deterministic_across_runs(self):
        samples = [s(f"s{i}", f"A{i:02d}", "2025-03-01") for i in range(12)]
        groups, spg, ev = build_eval(samples, [])
        r1 = allocate(groups, spg, ev, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=5)
        r2 = allocate(groups, spg, ev, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=5)
        self.assertEqual(r1.assignment, r2.assignment)


class RatioNormalizationTest(unittest.TestCase):
    def test_normalizes_to_one(self):
        out = normalize_ratios({"train": 70, "val": 15, "test": 15})
        self.assertAlmostEqual(sum(out.values()), 1.0)
        self.assertAlmostEqual(out["train"], 0.7)

    def test_all_zero_becomes_equal_thirds(self):
        out = normalize_ratios({"train": 0, "val": 0, "test": 0})
        self.assertEqual(out, {"train": 1 / 3, "val": 1 / 3, "test": 1 / 3})


if __name__ == "__main__":
    unittest.main()
