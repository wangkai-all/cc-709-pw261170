"""样本与关系的数据模型、CSV 读写及内容哈希。

所有哈希都基于规范化文本（统一 ``\\r\\n`` 为 ``\\n``），保证同一份数据
在不同平台上得到相同的版本指纹，从而支撑可复现划分。
"""
from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass, field
from datetime import date

SAMPLE_REQUIRED = ("sample_id", "subject_id", "event_date")
RELATION_REQUIRED = ("rel_id", "src_subject", "dst_subject", "rel_type")


class DatasetError(ValueError):
    """输入数据不合法（缺列、重复 ID、日期格式错误等）。"""


@dataclass(frozen=True)
class Sample:
    sample_id: str
    subject_id: str
    event_date: str  # ISO 8601 YYYY-MM-DD
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Relation:
    rel_id: str
    src_subject: str
    dst_subject: str
    rel_type: str
    active: bool = True


@dataclass
class Dataset:
    name: str
    samples: list[Sample]
    relations: list[Relation]
    sample_version: str = "samples-v1"
    relation_version: str = "relations-v1"

    def subject_ids(self) -> set[str]:
        return {s.subject_id for s in self.samples}


def normalize_text(text: str) -> str:
    """统一换行符，避免平台差异污染哈希。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _parse_active(raw: str | None) -> bool:
    if raw is None or raw == "":
        return True
    return raw.strip().lower() in ("1", "true", "yes", "y", "t")


def load_samples_csv(text: str) -> list[Sample]:
    """解析样本 CSV，必需列：sample_id, subject_id, event_date。

    其余列作为透传属性保留，最终原样写入产出清单。
    """
    reader = csv.DictReader(io.StringIO(normalize_text(text)))
    if reader.fieldnames is None:
        raise DatasetError("样本 CSV 为空或缺少表头")
    missing = [c for c in SAMPLE_REQUIRED if c not in reader.fieldnames]
    if missing:
        raise DatasetError(f"样本 CSV 缺少必需列: {', '.join(missing)}")

    samples: list[Sample] = []
    seen: set[str] = set()
    for line_no, row in enumerate(reader, start=2):
        values = {k: (v or "").strip() for k, v in row.items() if k is not None}
        sid = values.get("sample_id", "")
        if not sid:
            raise DatasetError(f"第 {line_no} 行 sample_id 为空")
        if sid in seen:
            raise DatasetError(f"样本 ID 重复: {sid}（第 {line_no} 行）")
        subject = values.get("subject_id", "")
        if not subject:
            raise DatasetError(f"样本 {sid} 的 subject_id 为空")
        event_date = values.get("event_date", "")
        try:
            date.fromisoformat(event_date)
        except ValueError as exc:
            raise DatasetError(
                f"样本 {sid} 的 event_date 不是 YYYY-MM-DD: {event_date!r}"
            ) from exc
        seen.add(sid)
        attrs = {
            k: values[k]
            for k in sorted(values)
            if k not in SAMPLE_REQUIRED
        }
        samples.append(Sample(sid, subject, event_date, attrs))
    if not samples:
        raise DatasetError("样本 CSV 没有任何数据行")
    return samples


def load_relations_csv(text: str) -> list[Relation]:
    """解析关系 CSV，必需列：rel_id, src_subject, dst_subject, rel_type。

    可选列 active（缺省视为 true），撤回关系即把 active 置为 false。
    """
    reader = csv.DictReader(io.StringIO(normalize_text(text)))
    if reader.fieldnames is None:
        raise DatasetError("关系 CSV 为空或缺少表头")
    missing = [c for c in RELATION_REQUIRED if c not in reader.fieldnames]
    if missing:
        raise DatasetError(f"关系 CSV 缺少必需列: {', '.join(missing)}")

    relations: list[Relation] = []
    seen: set[str] = set()
    for line_no, row in enumerate(reader, start=2):
        values = {k: (v or "").strip() for k, v in row.items() if k is not None}
        rid = values.get("rel_id", "")
        if not rid:
            raise DatasetError(f"第 {line_no} 行 rel_id 为空")
        if rid in seen:
            raise DatasetError(f"关系 ID 重复: {rid}（第 {line_no} 行）")
        seen.add(rid)
        if not values.get("src_subject") or not values.get("dst_subject"):
            raise DatasetError(f"关系 {rid} 的主体端点不能为空")
        if not values.get("rel_type"):
            raise DatasetError(f"关系 {rid} 的 rel_type 不能为空")
        relations.append(
            Relation(
                rel_id=rid,
                src_subject=values["src_subject"],
                dst_subject=values["dst_subject"],
                rel_type=values["rel_type"],
                active=_parse_active(values.get("active")),
            )
        )
    return relations


def dump_samples_csv(samples: list[Sample]) -> str:
    """确定性输出：按 sample_id 排序，属性列按列名排序。"""
    ordered = sorted(samples, key=lambda s: s.sample_id)
    attr_cols = sorted({k for s in ordered for k in s.attributes})
    cols = list(SAMPLE_REQUIRED) + attr_cols
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    writer.writeheader()
    for s in ordered:
        row = {
            "sample_id": s.sample_id,
            "subject_id": s.subject_id,
            "event_date": s.event_date,
            **{c: s.attributes.get(c, "") for c in attr_cols},
        }
        writer.writerow(row)
    return buf.getvalue()


def dump_relations_csv(relations: list[Relation]) -> str:
    """确定性输出：按 rel_id 排序。"""
    ordered = sorted(relations, key=lambda r: r.rel_id)
    cols = ("rel_id", "src_subject", "dst_subject", "rel_type", "active")
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    writer.writeheader()
    for r in ordered:
        writer.writerow(
            {
                "rel_id": r.rel_id,
                "src_subject": r.src_subject,
                "dst_subject": r.dst_subject,
                "rel_type": r.rel_type,
                "active": "true" if r.active else "false",
            }
        )
    return buf.getvalue()
