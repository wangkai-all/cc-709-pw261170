#!/usr/bin/env python3
"""端到端跑通内置示例，把实际清单与分析报告落盘到 examples/output/。

产出（均为确定性内容，重复运行逐字节一致）：

    examples/output/
      v1_relations-v1/
        train.csv val.csv test.csv quarantine.csv
        groups.json            连通组与组级分配理由
        allocation_report.md   目标/实际比例、偏差与大组解释
        freeze.json            冻结配置与内容哈希
      v2_relations-v2/         迟到关系修订后的候选版（v1 文件原样保留）
        ... 同上
      leakage_report.json      跨集合冲突（潜在泄漏）
      comparison.md            新旧版本逐样本对比
      verify.py                独立小规模穷举校验（可单独运行）

用法：python examples/run_example.py [--rule shift|isolate] [--out examples/output]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from splitter.examples import (  # noqa: E402
    LATE_REVISION,
    TRAIN_CUTOFF,
    VAL_CUTOFF,
    build_example_dataset,
    example_config,
)
from splitter.manifest import SplitConfig, generate_manifests  # noqa: E402
from splitter.revision import (  # noqa: E402
    analyze_leakage,
    compare_generations,
    merge_subjects,
)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def save_generation_bundle(out_dir: Path, gen) -> None:
    for split, csv_text in gen.manifest_csv.items():
        write_text(out_dir / f"{split}.csv", csv_text)
    write_text(out_dir / "quarantine.csv", gen.quarantine_csv)
    write_json(
        out_dir / "groups.json",
        [
            {
                "group_id": g["group_id"],
                "subjects": g["subjects"],
                "edge_ids": g["edge_ids"],
                "subject_count": g["subject_count"],
                "sample_count": g["sample_count"],
                "min_date": g["min_date"],
                "max_date": g["max_date"],
                "eligible_splits": g["eligible_splits"],
                "crosses_boundary": g["crosses_boundary"],
                "assigned_split": g["assigned_split"],
                "quarantined": g["quarantined"],
                "reason": g["reason"],
            }
            for g in gen.groups
        ],
    )
    write_json(
        out_dir / "freeze.json",
        {
            "freeze_id": gen.freeze_id,
            "generation_id": gen.generation_id,
            "sample_version": gen.sample_version,
            "relation_version": gen.relation_version,
            "sample_hash": gen.sample_hash,
            "relation_hash": gen.relation_hash,
            "config": gen.config,
        },
    )
    # 时间戳是运行元数据，不进入冻结文件，保证重复运行逐字节一致
    write_json(out_dir / "generated_at.json", {"created_at": gen.created_at})
    alloc = gen.allocation
    lines = [
        f"# 分配报告 {gen.generation_id}",
        "",
        f"- freeze_id: `{gen.freeze_id}`",
        f"- 样本版/关系版: `{gen.sample_version}` / `{gen.relation_version}`",
        f"- 策略: **{alloc['strategy']}**"
        + (f"（贪心 SSE={alloc['greedy_sse']:.2f}）" if alloc["greedy_sse"] is not None else ""),
        f"- 最优 SSE: {alloc['sse']:.2f}",
        "",
        "## 目标 vs 实际",
        "",
        "| 集合 | 目标比例 | 实际比例 | 目标样本数 | 实际样本数 | 偏差(样本) | 主体数 |",
        "|---|---|---|---|---|---|---|",
    ]
    for sp in ("train", "val", "test"):
        lines.append(
            f"| {sp} | {alloc['target_ratios'][sp]:.2%} | "
            f"{alloc['actual_ratios'][sp]:.2%} | {alloc['target_samples'][sp]:.1f} | "
            f"{alloc['split_samples'][sp]} | {alloc['deviation'][sp]:+.1f} | "
            f"{alloc['split_subjects'][sp]} |"
        )
    lines += ["", "## 为什么比例偏离目标（组级解释）", ""]
    lines += [f"- {note}" for note in alloc["notes"]]
    lines += [
        "",
        "## 时间约束",
        f"- train_cutoff={gen.config['train_cutoff']}（含当天），"
        f"val_cutoff={gen.config['val_cutoff']}（含当天）",
        f"- 跨边界规则: {gen.config['cross_boundary_rule']}（绝不拆组）",
        f"- 隔离样本数: {len(gen.quarantine)}",
        "",
        "## 连通组明细",
        "",
        "| 组 | 主体 | 样本 | 日期范围 | 跨边界 | 去向 | 理由 |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in gen.groups:
        lines.append(
            f"| `{g['group_id']}` | {g['subject_count']} | {g['sample_count']} | "
            f"{g['min_date']}~{g['max_date']} | {'是' if g['crosses_boundary'] else '否'} | "
            f"{g['assigned_split'] or ('quarantine' if g['quarantined'] else '-')} | "
            f"{g['reason'].replace('|', '/')} |"
        )
    write_text(out_dir / "allocation_report.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成内置示例的实际清单")
    parser.add_argument("--rule", choices=["shift", "isolate"], default="shift")
    parser.add_argument("--out", default=str(ROOT / "examples" / "output"))
    args = parser.parse_args()

    out_root = Path(args.out)
    config = SplitConfig(**{**example_config(args.rule), "cross_boundary_rule": args.rule})

    # ---------- v1：初版关系 ----------
    dataset_v1 = build_example_dataset()
    gen_v1 = generate_manifests(dataset_v1, config)
    v1_dir = out_root / f"v1_{gen_v1.relation_version}_{args.rule}"
    save_generation_bundle(v1_dir, gen_v1)

    # 可复现性自检：同配置重新生成必须逐字节一致
    again = generate_manifests(dataset_v1, config)
    assert again.freeze_id == gen_v1.freeze_id
    assert again.manifest_csv == gen_v1.manifest_csv

    # ---------- v2：迟到的关系修订（C04 与 S01 实为同一人） ----------
    dataset_v2 = merge_subjects(
        dataset_v1,
        LATE_REVISION["subject_a"],
        LATE_REVISION["subject_b"],
        LATE_REVISION["rel_type"],
    )
    report = analyze_leakage(gen_v1, dataset_v1, dataset_v2, config)
    gen_v2 = generate_manifests(dataset_v2, config)
    v2_dir = out_root / f"v2_{gen_v2.relation_version}_{args.rule}"
    save_generation_bundle(v2_dir, gen_v2)
    write_json(out_root / "leakage_report.json", report)

    cmp = compare_generations(gen_v1, gen_v2)
    cmp_lines = [
        "# 新旧版本对比",
        "",
        f"- 旧版 `{cmp['old_generation_id']}`（{cmp['old_relation_version']}）",
        f"- 新版 `{cmp['new_generation_id']}`（{cmp['new_relation_version']}）",
        f"- 跨集合移动样本数: **{cmp['samples_moved_count']}**",
        "",
        "| 集合 | 目标 | 旧实际 | 新实际 |",
        "|---|---|---|---|",
    ]
    for sp in ("train", "val", "test"):
        cmp_lines.append(
            f"| {sp} | {cmp['ratio_table']['target'][sp]:.2%} | "
            f"{cmp['ratio_table']['old_actual'][sp]:.2%} "
            f"({cmp['sample_count_table']['old'][sp]}) | "
            f"{cmp['ratio_table']['new_actual'][sp]:.2%} "
            f"({cmp['sample_count_table']['new'][sp]}) |"
        )
    cmp_lines += ["", "## 移动明细", "", "| sample_id | from | to |", "|---|---|---|"]
    cmp_lines += [f"| {m['sample_id']} | {m['from']} | {m['to']} |" for m in cmp["samples_moved"]]
    write_text(out_root / "comparison.md", "\n".join(cmp_lines) + "\n")

    print(f"[ok] v1 清单与报告 -> {v1_dir}")
    print(f"[ok] v2 候选清单   -> {v2_dir}（v1 文件未改动）")
    print(f"[ok] 泄漏报告      -> {out_root / 'leakage_report.json'}")
    print(f"[ok] 版本对比      -> {out_root / 'comparison.md'}")
    print(f"[ok] 时间边界      train<={TRAIN_CUTOFF}, val<={VAL_CUTOFF}, 规则={args.rule}")
    print(
        "[ok] 实际比例      "
        + ", ".join(f"{sp}={gen_v1.allocation['actual_ratios'][sp]:.2%}" for sp in ("train", "val", "test"))
    )
    print(f"[ok] 泄漏冲突      {len(report['cross_split_conflicts'])} 处，"
          f"涉及 {report['affected_sample_count']} 个样本")


if __name__ == "__main__":
    main()
