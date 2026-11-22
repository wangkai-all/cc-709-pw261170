"""命令行入口：python -m splitter --host 127.0.0.1 --port 8000"""
from __future__ import annotations

import argparse

from .api import run


def main() -> None:
    parser = argparse.ArgumentParser(description="训练数据湖可重复数据集划分工具")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--data",
        default=None,
        help="数据存储目录（默认 ./data，也可用环境变量 SPLITTER_DATA）",
    )
    args = parser.parse_args()
    run(host=args.host, port=args.port, data_dir=args.data)


if __name__ == "__main__":
    main()
