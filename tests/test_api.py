"""API 端到端测试：真实启动 ThreadingHTTPServer，走完整 HTTP 工作流。"""
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from splitter.api import make_handler
from splitter.examples import example_config
from splitter.store import Store


class ApiEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.store = Store(cls.tmp.name)
        handler = make_handler(cls.store)
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)
        cls.tmp.cleanup()

    def _url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def _request(self, method, path, payload=None):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            self._url(path),
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def _get_csv(self, path):
        with urllib.request.urlopen(self._url(path), timeout=10) as resp:
            return resp.status, resp.headers.get("Content-Disposition"), resp.read().decode("utf-8")

    def test_full_workflow(self):
        # 1. 健康检查与静态首页
        status, body = self._request("GET", "/api/health")
        self.assertEqual((status, body["status"]), (200, "ok"))
        with urllib.request.urlopen(self._url("/"), timeout=10) as resp:
            self.assertIn("数据集划分工作台", resp.read().decode("utf-8"))

        # 2. 载入内置示例
        status, body = self._request("POST", "/api/datasets/example", {"name": "lake_demo"})
        self.assertEqual(status, 201)
        self.assertEqual(body["samples"], 69)

        # 重复创建要被拒绝
        status, body = self._request("POST", "/api/datasets/example", {"name": "lake_demo"})
        self.assertEqual(status, 409)

        # 3. 数据集详情
        status, body = self._request("GET", "/api/datasets/lake_demo")
        self.assertEqual(status, 200)
        self.assertEqual(body["relation_version"], "relations-v1")

        # 4. 生成划分
        cfg = example_config()
        status, gen = self._request("POST", "/api/generate",
                                    {"dataset": "lake_demo", "config": cfg})
        self.assertEqual(status, 201)
        gen_id = gen["generation_id"]
        self.assertEqual(len(gen["groups"]), 12)
        # 同配置再次生成 -> 同 generation（幂等可复现）
        status, gen_again = self._request("POST", "/api/generate",
                                          {"dataset": "lake_demo", "config": cfg})
        self.assertEqual(gen_again["generation_id"], gen_id)

        # 5. 下载三份 CSV
        for sp in ("train", "val", "test"):
            status, disposition, text = self._get_csv(
                f"/api/generations/{gen_id}/manifest/{sp}"
            )
            self.assertEqual(status, 200)
            self.assertIn("attachment", disposition)
            self.assertTrue(text.startswith("sample_id,subject_id,event_date"))

        # 6. 迟到关系修订：合并 C04/S01 -> 候选版 + 泄漏
        status, body = self._request("POST", "/api/revise/merge", {
            "dataset": "lake_demo", "subject_a": "C04", "subject_b": "S01",
            "rel_type": "same_as",
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["relation_version"], "relations-v2")

        status, body = self._request("POST", "/api/revise/candidate", {
            "dataset": "lake_demo", "old_generation_id": gen_id, "config": cfg,
        })
        self.assertEqual(status, 201)
        self.assertTrue(body["leakage"]["leakage"])
        candidate_id = body["candidate"]["generation_id"]
        self.assertNotEqual(candidate_id, gen_id)
        # 旧 generation 内容未被修改
        status, old = self._request("GET", f"/api/generations/{gen_id}")
        self.assertEqual(old["relation_version"], "relations-v1")

        # 7. 版本对比
        status, cmp = self._request(
            "GET", f"/api/compare?old={gen_id}&new={candidate_id}"
        )
        self.assertEqual(status, 200)
        self.assertGreater(cmp["samples_moved_count"], 0)

        # 8. 发布两个版本
        status, pub1 = self._request("POST", "/api/publish", {"generation_id": gen_id})
        self.assertEqual(pub1["published_as"], "v1")
        status, pub2 = self._request("POST", "/api/publish",
                                     {"generation_id": candidate_id})
        self.assertEqual(pub2["published_as"], "v2")
        status, pubs = self._request("GET", "/api/publications?dataset=lake_demo")
        self.assertEqual([p["published_as"] for p in pubs["publications"]], ["v1", "v2"])

        # 9. 撤回关系
        status, body = self._request("POST", "/api/revise/withdraw", {
            "dataset": "lake_demo", "rel_id": "rel_c03_c04",
        })
        self.assertEqual(status, 200)

    def test_validation_errors_are_400(self):
        status, body = self._request("POST", "/api/datasets", {
            "name": "bad",
            "samples_csv": "sample_id,subject_id\ns1,A\n",
            "relations_csv": "rel_id,src_subject,dst_subject,rel_type\n",
        })
        self.assertEqual(status, 400)
        self.assertIn("缺少必需列", body["error"])

    def test_unknown_route_404(self):
        status, _ = self._request("GET", "/api/nope")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
