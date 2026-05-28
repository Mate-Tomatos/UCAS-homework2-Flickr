"""检查 Flickr30k 标注与图片文件是否匹配。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        命令行参数对象。
    """
    parser = argparse.ArgumentParser(description="检查 Flickr30k 本地数据完整性。")
    parser.add_argument(
        "--annotations-csv",
        type=Path,
        default=Path("datasets/flickr30k_hf/flickr_annotations_30k.csv"),
        help="HF Flickr30k 标注 CSV 路径。",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="解压后的 Flickr30k 图片根目录。",
    )
    parser.add_argument(
        "--sample-missing",
        type=int,
        default=20,
        help="最多打印多少个缺失文件名。",
    )
    return parser.parse_args()


def build_image_index(image_root: Path) -> set[str]:
    """构建图片文件名索引。

    Args:
        image_root: 图片根目录。

    Returns:
        所有 jpg/jpeg 图片的文件名集合。
    """
    suffixes = {".jpg", ".jpeg"}
    return {
        path.name
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    }


def main() -> None:
    """执行数据完整性检查。"""
    args = parse_args()
    image_names = build_image_index(args.image_root)
    split_counts: Counter[str] = Counter()
    caption_counts: Counter[str] = Counter()
    missing: list[str] = []

    with args.annotations_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split = row["split"]
            filename = row["filename"]
            captions = json.loads(row["raw"])
            split_counts[split] += 1
            caption_counts[split] += len(captions)
            if filename not in image_names:
                missing.append(filename)

    print("标注图片数:", sum(split_counts.values()))
    print("标注 caption 数:", sum(caption_counts.values()))
    print("本地图片数:", len(image_names))
    print("split 图片数:", dict(sorted(split_counts.items())))
    print("split caption 数:", dict(sorted(caption_counts.items())))
    print("缺失图片数:", len(missing))
    for filename in missing[: args.sample_missing]:
        print("缺失:", filename)


if __name__ == "__main__":
    main()
