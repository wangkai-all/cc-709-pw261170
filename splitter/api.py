"""HTTP 接口：仅依赖标准库，把核心能力暴露给单页前端。

所有写操作都落盘到 :class:`~splitter.store.Store`，已发布/已生成的文件
永不覆盖，从存储层面保证旧清单不被悄悄改掉。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .dataset import DatasetError
from .examples import LATE_REVISION, build_example_dataset, example_config
from .manifest import Generation, SplitConfig, generate_manifests
from .revision import (
    analyze_leakage,
    compare_generations,
    merge_subjects,
    withdraw_relation,
)
from .store import Store

STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def config_from_payload(payload: dict) -> SplitConfig:
    allowed = {
        "seed",
        "target_ratios",
        "train_cutoff",
        "val_cutoff",
        "cross_boundary_rule",
        "relation_types",
        "exhaustive_limit",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"未知配置项: {sorted(unknown)}")
    data = dict(payload)
    if "seed" in data:
        try:
            data["seed"] = int(data["seed"])
        except (TypeError, ValueError) as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "seed 必须是整数") from exc
    if "exhaustive_limit" in data:
        try:
            data["exhaustive_limit"] = int(data["exhaustive_limit"])
        except (TypeError, ValueError) as exc:
            raise ApiError(
                HTTPStatus.BAD_REQUEST, "exhaustive_limit 必须是整数"
            ) from exc
    return SplitConfig(**data)


def generation_dict(generation: Generation, leakage: dict | None = None) -> dict:
    payload = asdict(generation)
    if leakage is not None:
        payload["leakage_report"] = leakage
    return payload


def make_handler(store: Store) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "LakeSplitter/1.0"

        def log_message(self, fmt: str, *args: object) -> None:  # 安静一点
            if os.environ.get("SPLITTER_HTTP_LOG"):
                super().log_message(fmt, *args)

        # ------------------------------------------------------------
        # 基础收发
        # ------------------------------------------------------------

        def _send_json(self, status: int, payload: object) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            if not length:
                return {}
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ApiError(HTTPStatus.BAD_REQUEST, f"请求体不是合法 JSON: {exc}")
            if not isinstance(payload, dict):
                raise ApiError(HTTPStatus.BAD_REQUEST, "请求体必须是 JSON 对象")
            return payload

        def _send_file(self, path: Path, content_type: str, download_name: str | None = None) -> None:
            if not path.is_file():
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"文件不存在: {path.name}"})
                return
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if download_name:
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="{download_name}"',
                )
            self.end_headers()
            self.wfile.write(body)

        def _require(self, payload: dict, key: str) -> object:
            if key not in payload or payload[key] in (None, ""):
                raise ApiError(HTTPStatus.BAD_REQUEST, f"缺少必填参数: {key}")
            return payload[key]

        # ------------------------------------------------------------
        # 路由
        # ------------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            try:
                self._route_get()
            except ApiError as exc:
                self._send_json(exc.status, {"error": exc.message})
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

        def do_POST(self) -> None:  # noqa: N802
            try:
                self._route_post()
            except ApiError as exc:
                self._send_json(exc.status, {"error": exc.message})
            except (DatasetError, ValueError, KeyError, FileExistsError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

        def _route_get(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            qs = parse_qs(parsed.query)

            if path == "/api/health":
                self._send_json(HTTPStatus.OK, {"status": "ok"})
                return
            if path == "/api/datasets":
                self._send_json(HTTPStatus.OK, {"datasets": store.list_datasets()})
                return
            if path == "/api/generations":
                dataset_name = qs.get("dataset", [None])[0]
                self._send_json(
                    HTTPStatus.OK,
                    {"generations": store.list_generations(dataset_name)},
                )
                return

            parts = [p for p in path.split("/") if p]
            # /api/datasets/<name>
            if len(parts) == 3 and parts[:2] == ["api", "datasets"]:
                dataset = store.load_dataset(parts[2])
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "name": dataset.name,
                        "sample_version": dataset.sample_version,
                        "relation_version": dataset.relation_version,
                        "sample_count": len(dataset.samples),
                        "relation_count": len(dataset.relations),
                        "samples": [asdict(s) for s in dataset.samples],
                        "relations": [asdict(r) for r in dataset.relations],
                    },
                )
                return
            # /api/generations/<id>
            if len(parts) == 3 and parts[:2] == ["api", "generations"]:
                generation = store.load_generation(parts[2])
                leakage = store.load_leakage_report(parts[2])
                self._send_json(
                    HTTPStatus.OK, generation_dict(generation, leakage)
                )
                return
            # /api/generations/<id>/manifest/<split>
            if (
                len(parts) == 5
                and parts[:2] == ["api", "generations"]
                and parts[3] == "manifest"
            ):
                generation = store.load_generation(parts[2])
                split = parts[4]
                if split == "quarantine":
                    text = generation.quarantine_csv
                    fname = f"{generation.generation_id}_quarantine.csv"
                elif split in generation.manifest_csv:
                    text = generation.manifest_csv[split]
                    fname = f"{generation.generation_id}_{split}.csv"
                else:
                    raise ApiError(HTTPStatus.NOT_FOUND, f"未知集合: {split}")
                body = text.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{fname}"'
                )
                self.end_headers()
                self.wfile.write(body)
                return
            # /api/publications?dataset=<name>
            if path == "/api/publications":
                dataset_name = self._require_qs(qs, "dataset")
                self._send_json(
                    HTTPStatus.OK,
                    {"publications": store.list_publications(dataset_name)},
                )
                return
            # /api/compare?old=<id>&new=<id>
            if path == "/api/compare":
                old = store.load_generation(self._require_qs(qs, "old"))
                new = store.load_generation(self._require_qs(qs, "new"))
                self._send_json(HTTPStatus.OK, compare_generations(old, new))
                return
            # 静态资源
            if path == "/" or not path.startswith("/api"):
                self._serve_static(path)
                return
            raise ApiError(HTTPStatus.NOT_FOUND, f"未知路径: {path}")

        @staticmethod
        def _require_qs(qs: dict[str, list[str]], key: str) -> str:
            value = qs.get(key, [""])[0]
            if not value:
                raise ApiError(HTTPStatus.BAD_REQUEST, f"缺少查询参数: {key}")
            return value

        def _serve_static(self, path: str) -> None:
            rel = "index.html" if path == "/" else path.lstrip("/")
            target = (STATIC_DIR / rel).resolve()
            if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
                target = STATIC_DIR / "index.html"
            content_types = {
                ".html": "text/html; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8",
            }
            self._send_file(target, content_types.get(target.suffix, "application/octet-stream"))

        def _route_post(self) -> None:
            path = urlparse(self.path).path.rstrip("/")
            payload = self._read_json()

            if path == "/api/datasets":
                self._create_dataset(payload)
                return
            if path == "/api/datasets/example":
                self._create_example(payload)
                return
            if path == "/api/generate":
                self._generate(payload)
                return
            if path == "/api/revise/withdraw":
                self._withdraw(payload)
                return
            if path == "/api/revise/merge":
                self._merge(payload)
                return
            if path == "/api/revise/candidate":
                self._candidate(payload)
                return
            if path == "/api/publish":
                result = store.publish(str(self._require(payload, "generation_id")))
                self._send_json(HTTPStatus.OK, result)
                return
            raise ApiError(HTTPStatus.NOT_FOUND, f"未知路径: {path}")

        # ------------------------------------------------------------
        # 业务动作
        # ------------------------------------------------------------

        def _create_dataset(self, payload: dict) -> None:
            name = str(self._require(payload, "name"))
            dataset = store.create_dataset(
                name=name,
                samples_csv=str(payload.get("samples_csv", "")),
                relations_csv=str(payload.get("relations_csv", "")),
                sample_version=str(payload.get("sample_version", "samples-v1")),
                relation_version=str(payload.get("relation_version", "relations-v1")),
            )
            self._send_json(
                HTTPStatus.CREATED,
                {"name": dataset.name, "samples": len(dataset.samples),
                 "relations": len(dataset.relations)},
            )

        def _create_example(self, payload: dict) -> None:
            name = str(payload.get("name", "lake_demo"))
            dataset = build_example_dataset()
            dataset.name = name
            from .dataset import dump_relations_csv, dump_samples_csv

            try:
                store.create_dataset(
                    name=name,
                    samples_csv=dump_samples_csv(dataset.samples),
                    relations_csv=dump_relations_csv(dataset.relations),
                    sample_version=dataset.sample_version,
                    relation_version=dataset.relation_version,
                )
            except FileExistsError:
                raise ApiError(
                    HTTPStatus.CONFLICT, f"数据集 {name} 已存在，请换个名字"
                )
            self._send_json(
                HTTPStatus.CREATED,
                {
                    "name": name,
                    "samples": len(dataset.samples),
                    "relations": len(dataset.relations),
                    "late_revision_hint": LATE_REVISION,
                    "example_config": example_config(),
                },
            )

        def _generate(self, payload: dict) -> None:
            name = str(self._require(payload, "dataset"))
            config = config_from_payload(payload.get("config", {}))
            dataset = store.load_dataset(name)
            generation = generate_manifests(dataset, config)
            store.save_generation(generation)
            self._send_json(HTTPStatus.CREATED, generation_dict(generation))

        def _withdraw(self, payload: dict) -> None:
            name = str(self._require(payload, "dataset"))
            rel_id = str(self._require(payload, "rel_id"))
            dataset = store.load_dataset(name)
            revised = withdraw_relation(dataset, rel_id)
            store.save_dataset_revision(revised)
            self._send_json(
                HTTPStatus.OK,
                {"dataset": name, "relation_version": revised.relation_version,
                 "withdrawn": rel_id},
            )

        def _merge(self, payload: dict) -> None:
            name = str(self._require(payload, "dataset"))
            subject_a = str(self._require(payload, "subject_a"))
            subject_b = str(self._require(payload, "subject_b"))
            rel_type = str(payload.get("rel_type", "same_as"))
            dataset = store.load_dataset(name)
            revised = merge_subjects(dataset, subject_a, subject_b, rel_type)
            store.save_dataset_revision(revised)
            self._send_json(
                HTTPStatus.OK,
                {"dataset": name, "relation_version": revised.relation_version,
                 "added_relation": f"rel_merge_{'__'.join(sorted((subject_a, subject_b)))}"},
            )

        def _candidate(self, payload: dict) -> None:
            """修订后生成候选版：旧 Generation 原样保留，只附泄漏报告。

            旧版生成对应的数据快照按它冻结的 relation_version 取回，
            当前数据集作为修订版，两者都不会被覆盖。
            """
            name = str(self._require(payload, "dataset"))
            old_id = str(self._require(payload, "old_generation_id"))
            config = config_from_payload(payload.get("config", {}))
            old = store.load_generation(old_id)
            old_dataset = store.load_dataset(
                name,
                sample_version=old.sample_version,
                relation_version=old.relation_version,
            )
            current = store.load_dataset(name)
            report = analyze_leakage(old, old_dataset, current, config)
            candidate = generate_manifests(current, config)
            store.save_generation(candidate)
            store.save_leakage_report(candidate.generation_id, report)
            self._send_json(
                HTTPStatus.CREATED,
                {
                    "candidate": generation_dict(candidate, report),
                    "leakage": report,
                    "old_generation": {
                        "generation_id": old.generation_id,
                        "freeze_id": old.freeze_id,
                        "published": old.published,
                    },
                },
            )

    return Handler


def run(host: str = "127.0.0.1", port: int = 8000, data_dir: str | None = None) -> None:
    data_dir = data_dir or os.environ.get("SPLITTER_DATA", "data")
    store = Store(data_dir)
    handler = make_handler(store)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"数据集划分工具运行中: http://{host}:{port}  数据目录: {Path(data_dir).resolve()}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()
