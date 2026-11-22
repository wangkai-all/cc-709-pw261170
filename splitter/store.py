"""基于文件系统的不可变快照存储（仅用标准库，无需数据库）。

目录结构::

    data/
      datasets/<dataset_id>/
        meta.json                 名称、当前版本指针
        versions/<version>/{samples.csv,relations.csv,meta.json}
      generations/<generation_id>/generation.json
      publications/
        <dataset_id>/v1/generation.json     已发布版本（永不覆盖）
        <dataset_id>/v2/...
      leakages/<id>.json                     候选版附带的泄漏报告

数据集每次修订都产生新版本目录；生成结果与发布版本均为不可变 JSON，
"原清单保留、只标注泄漏"在物理上由"不覆盖旧文件"保证。
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

from .dataset import (
    Dataset,
    dump_relations_csv,
    dump_samples_csv,
    load_relations_csv,
    load_samples_csv,
)
from .manifest import Generation


def _json_default(obj: object) -> object:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"不可序列化的对象: {type(obj)!r}")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Store:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 数据集
    # ------------------------------------------------------------------

    def create_dataset(
        self,
        name: str,
        samples_csv: str,
        relations_csv: str,
        sample_version: str = "samples-v1",
        relation_version: str = "relations-v1",
    ) -> Dataset:
        samples = load_samples_csv(samples_csv)
        relations = load_relations_csv(relations_csv)
        dataset = Dataset(
            name=name,
            samples=samples,
            relations=relations,
            sample_version=sample_version,
            relation_version=relation_version,
        )
        dataset_id = self._dataset_id(name)
        base = self.root / "datasets" / dataset_id
        if base.exists():
            raise FileExistsError(f"数据集 {name} 已存在: {base}")
        self._persist_dataset(dataset_id, dataset)
        write_json(
            base / "meta.json",
            {
                "dataset_id": dataset_id,
                "name": name,
                "current_sample_version": sample_version,
                "current_relation_version": relation_version,
            },
        )
        return dataset

    @staticmethod
    def _dataset_id(name: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        return safe or "dataset"

    def _persist_dataset(self, dataset_id: str, dataset: Dataset) -> None:
        version_dir = (
            self.root
            / "datasets"
            / dataset_id
            / "versions"
            / f"{dataset.sample_version}__{dataset.relation_version}"
        )
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "samples.csv").write_text(
            dump_samples_csv(dataset.samples), encoding="utf-8"
        )
        (version_dir / "relations.csv").write_text(
            dump_relations_csv(dataset.relations), encoding="utf-8"
        )
        write_json(
            version_dir / "meta.json",
            {
                "sample_version": dataset.sample_version,
                "relation_version": dataset.relation_version,
                "sample_count": len(dataset.samples),
                "relation_count": len(dataset.relations),
            },
        )

    def save_dataset_revision(self, dataset: Dataset) -> None:
        """修订后的 Dataset 写入新版本目录，并更新当前版本指针。"""
        dataset_id = self._dataset_id(dataset.name)
        self._persist_dataset(dataset_id, dataset)
        meta_path = self.root / "datasets" / dataset_id / "meta.json"
        meta = read_json(meta_path)
        meta["current_sample_version"] = dataset.sample_version
        meta["current_relation_version"] = dataset.relation_version
        write_json(meta_path, meta)

    def load_dataset(
        self,
        name: str,
        sample_version: str | None = None,
        relation_version: str | None = None,
    ) -> Dataset:
        dataset_id = self._dataset_id(name)
        if sample_version is None or relation_version is None:
            meta = read_json(self.root / "datasets" / dataset_id / "meta.json")
            sample_version = sample_version or meta["current_sample_version"]
            relation_version = relation_version or meta["current_relation_version"]
        version_dir = (
            self.root
            / "datasets"
            / dataset_id
            / "versions"
            / f"{sample_version}__{relation_version}"
        )
        if not version_dir.exists():
            raise FileNotFoundError(
                f"数据集 {name} 的历史版本 {sample_version}__{relation_version} 不存在"
            )
        return Dataset(
            name=name,
            samples=load_samples_csv(
                (version_dir / "samples.csv").read_text(encoding="utf-8")
            ),
            relations=load_relations_csv(
                (version_dir / "relations.csv").read_text(encoding="utf-8")
            ),
            sample_version=sample_version,
            relation_version=relation_version,
        )

    def list_datasets(self) -> list[dict]:
        base = self.root / "datasets"
        if not base.exists():
            return []
        out = []
        for child in sorted(base.iterdir()):
            meta_path = child / "meta.json"
            if meta_path.exists():
                meta = read_json(meta_path)
                versions = sorted(
                    p.name for p in (child / "versions").iterdir() if p.is_dir()
                )
                meta["versions"] = versions
                out.append(meta)
        return out

    # ------------------------------------------------------------------
    # 生成（不可变）
    # ------------------------------------------------------------------

    def save_generation(self, generation: Generation) -> Path:
        path = (
            self.root / "generations" / generation.generation_id / "generation.json"
        )
        if path.exists():
            existing = read_json(path)
            if existing["freeze_id"] != generation.freeze_id:
                raise RuntimeError(
                    f"generation_id 冲突且 freeze_id 不同: {generation.generation_id}"
                )
            # 同 freeze_id 重复保存：内容必然等价，直接返回旧文件
            return path
        write_json(path, generation)
        return path

    def load_generation(self, generation_id: str) -> Generation:
        payload = read_json(
            self.root / "generations" / generation_id / "generation.json"
        )
        return self._generation_from_dict(payload)

    def list_generations(self, dataset_name: str | None = None) -> list[dict]:
        base = self.root / "generations"
        if not base.exists():
            return []
        items = []
        for child in sorted(base.iterdir()):
            payload = read_json(child / "generation.json")
            if dataset_name and payload["dataset_name"] != dataset_name:
                continue
            items.append(
                {
                    "generation_id": payload["generation_id"],
                    "freeze_id": payload["freeze_id"],
                    "dataset_name": payload["dataset_name"],
                    "created_at": payload["created_at"],
                    "sample_version": payload["sample_version"],
                    "relation_version": payload["relation_version"],
                    "published": payload.get("published", False),
                    "published_as": payload.get("published_as"),
                    "strategy": payload["allocation"]["strategy"],
                    "actual_ratios": payload["allocation"]["actual_ratios"],
                }
            )
        items.sort(key=lambda x: x["created_at"], reverse=True)
        return items

    @staticmethod
    def _generation_from_dict(payload: dict) -> Generation:
        return Generation(**payload)

    # ------------------------------------------------------------------
    # 发布（不可变，版本号只增不减）
    # ------------------------------------------------------------------

    def publish(self, generation_id: str) -> dict:
        gen_path = self.root / "generations" / generation_id / "generation.json"
        payload = read_json(gen_path)
        dataset_id = self._dataset_id(payload["dataset_name"])
        pub_root = self.root / "publications" / dataset_id
        existing_versions = (
            sorted(int(p.name[1:]) for p in pub_root.glob("v*") if p.is_dir())
            if pub_root.exists()
            else []
        )
        version_no = (existing_versions[-1] + 1) if existing_versions else 1
        version_name = f"v{version_no}"
        pub_dir = pub_root / version_name
        pub_dir.mkdir(parents=True)
        shutil.copy2(gen_path, pub_dir / "generation.json")

        payload["published"] = True
        payload["published_as"] = version_name
        write_json(gen_path, payload)
        return {"published_as": version_name, "generation_id": generation_id}

    def list_publications(self, dataset_name: str) -> list[dict]:
        dataset_id = self._dataset_id(dataset_name)
        pub_root = self.root / "publications" / dataset_id
        if not pub_root.exists():
            return []
        out = []
        for pub_dir in sorted(pub_root.glob("v*")):
            payload = read_json(pub_dir / "generation.json")
            out.append(
                {
                    "published_as": pub_dir.name,
                    "generation_id": payload["generation_id"],
                    "freeze_id": payload["freeze_id"],
                    "created_at": payload["created_at"],
                    "relation_version": payload["relation_version"],
                }
            )
        return out

    # ------------------------------------------------------------------
    # 泄漏报告（随候选版保存，不动旧清单）
    # ------------------------------------------------------------------

    def save_leakage_report(
        self, candidate_id: str, report: dict
    ) -> Path:
        path = self.root / "leakages" / f"{candidate_id}.json"
        write_json(path, report)
        return path

    def load_leakage_report(self, candidate_id: str) -> dict | None:
        path = self.root / "leakages" / f"{candidate_id}.json"
        return read_json(path) if path.exists() else None
