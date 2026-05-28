"""使用 CLIP 在 Flickr30k 上做零样本图文检索评测。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


@dataclass(frozen=True)
class FlickrItem:
    """Flickr30k 单张图片及其五条描述。"""

    filename: str
    captions: list[str]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        命令行参数对象。
    """
    parser = argparse.ArgumentParser(description="CLIP Flickr30k 零样本检索评测。")
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
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
        help="评测 split。",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP 模型名或本地路径。",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="文本/图片编码 batch size。")
    parser.add_argument("--max-images", type=int, default=0, help="调试时最多评测多少张图，0 表示全量。")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/clip_flickr30k_metrics.json"),
        help="评测指标输出路径。",
    )
    return parser.parse_args()


def load_items(annotations_csv: Path, split: str, max_images: int) -> list[FlickrItem]:
    """读取指定 split 的标注。

    Args:
        annotations_csv: Flickr30k 标注 CSV。
        split: 需要读取的 split。
        max_images: 最多读取图片数，0 表示不限制。

    Returns:
        评测样本列表。
    """
    items: list[FlickrItem] = []
    with annotations_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["split"] != split:
                continue
            captions = json.loads(row["raw"])
            items.append(FlickrItem(filename=row["filename"], captions=captions))
            if max_images > 0 and len(items) >= max_images:
                break
    return items


def build_image_paths(image_root: Path) -> dict[str, Path]:
    """建立文件名到图片路径的映射。

    Args:
        image_root: 图片根目录。

    Returns:
        图片文件名到实际路径的映射。
    """
    image_paths: dict[str, Path] = {}
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}:
            image_paths[path.name] = path
    return image_paths


def normalize(features: torch.Tensor) -> torch.Tensor:
    """对特征向量做 L2 归一化。

    Args:
        features: 待归一化特征。

    Returns:
        归一化后的特征。
    """
    return features / features.norm(dim=-1, keepdim=True)


def encode_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    items: list[FlickrItem],
    image_paths: dict[str, Path],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """批量编码图片。

    Args:
        model: CLIP 模型。
        processor: CLIP 预处理器。
        items: 评测样本。
        image_paths: 图片文件名到路径的映射。
        batch_size: batch size。
        device: 运行设备。

    Returns:
        图片特征矩阵。

    Raises:
        FileNotFoundError: 标注中的图片不在本地图片目录。
    """
    features: list[torch.Tensor] = []
    for start in range(0, len(items), batch_size):
        batch_items = items[start : start + batch_size]
        images: list[Image.Image] = []
        for item in batch_items:
            path = image_paths.get(item.filename)
            if path is None:
                raise FileNotFoundError(f"图片不存在: {item.filename}")
            images.append(Image.open(path).convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            batch_features = model.get_image_features(**inputs)
        features.append(normalize(batch_features).cpu())
        for image in images:
            image.close()
    return torch.cat(features, dim=0)


def encode_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    captions: list[str],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """批量编码文本。

    Args:
        model: CLIP 模型。
        processor: CLIP 预处理器。
        captions: caption 文本列表。
        batch_size: batch size。
        device: 运行设备。

    Returns:
        文本特征矩阵。
    """
    features: list[torch.Tensor] = []
    for start in range(0, len(captions), batch_size):
        batch = captions[start : start + batch_size]
        inputs = processor(
            text=batch,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            batch_features = model.get_text_features(**inputs)
        features.append(normalize(batch_features).cpu())
    return torch.cat(features, dim=0)


def recall_at(ranks: list[int], k: int) -> float:
    """计算 Recall@K。

    Args:
        ranks: 每个查询的正确结果排名，0 表示第一名。
        k: Recall 的 K 值。

    Returns:
        Recall@K 百分比。
    """
    hits = sum(rank < k for rank in ranks)
    return 100.0 * hits / len(ranks)


def compute_metrics(similarity: torch.Tensor, captions_per_image: int) -> dict[str, float]:
    """计算图搜文和文搜图召回率。

    Args:
        similarity: 图片到文本相似度矩阵，形状为图片数乘文本数。
        captions_per_image: 每张图对应的 caption 数。

    Returns:
        指标字典。
    """
    image_to_text_ranks: list[int] = []
    text_to_image_ranks: list[int] = []
    image_count = similarity.shape[0]
    caption_count = similarity.shape[1]

    sorted_text_indices = torch.argsort(similarity, dim=1, descending=True)
    for image_index in range(image_count):
        target_start = image_index * captions_per_image
        target_indices = set(range(target_start, target_start + captions_per_image))
        ranked = sorted_text_indices[image_index].tolist()
        image_to_text_ranks.append(min(ranked.index(index) for index in target_indices))

    sorted_image_indices = torch.argsort(similarity, dim=0, descending=True)
    for caption_index in range(caption_count):
        target_image = caption_index // captions_per_image
        ranked = sorted_image_indices[:, caption_index].tolist()
        text_to_image_ranks.append(ranked.index(target_image))

    metrics = {
        "image_to_text_r1": recall_at(image_to_text_ranks, 1),
        "image_to_text_r5": recall_at(image_to_text_ranks, 5),
        "image_to_text_r10": recall_at(image_to_text_ranks, 10),
        "text_to_image_r1": recall_at(text_to_image_ranks, 1),
        "text_to_image_r5": recall_at(text_to_image_ranks, 5),
        "text_to_image_r10": recall_at(text_to_image_ranks, 10),
    }
    metrics["rsum"] = sum(metrics.values())
    return metrics


def main() -> None:
    """运行 CLIP Flickr30k 零样本检索评测。"""
    args = parse_args()
    items = load_items(args.annotations_csv, args.split, args.max_images)
    if not items:
        raise ValueError(f"没有找到 split={args.split} 的样本。")
    image_paths = build_image_paths(args.image_root)
    captions = [caption for item in items for caption in item.captions]
    captions_per_image = len(items[0].captions)
    if any(len(item.captions) != captions_per_image for item in items):
        raise ValueError("每张图片的 caption 数不一致。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()

    image_features = encode_images(model, processor, items, image_paths, args.batch_size, device)
    text_features = encode_texts(model, processor, captions, args.batch_size, device)
    similarity = image_features @ text_features.T
    metrics = compute_metrics(similarity, captions_per_image)

    output = {
        "model_name": args.model_name,
        "split": args.split,
        "num_images": len(items),
        "num_captions": len(captions),
        "metrics": metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
