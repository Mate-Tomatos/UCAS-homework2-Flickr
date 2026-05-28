# 作业二：基于图神经网络的图文检索

本目录用于完成“基于图神经网络的搜索”作业。基础复现项目已克隆到
`VSRN/`，Flickr30k 标注与图片数据下载到 `datasets/`，加分项使用 CLIP 在
Flickr30k 上做零样本图文检索评测。

## 目录结构

- `VSRN/`: 官方 VSRN 代码，来源为 <https://github.com/KunpengLi1994/VSRN>。
- `datasets/flickr30k_hf/`: 从 Hugging Face `nlphuji/flickr30k` 下载的标注文件。
- `datasets/flickr30k_github_parts/`: 从 GitHub release 下载的 Flickr30k 图片分片。
- `scripts/inspect_flickr30k.py`: 检查图片与标注是否匹配。
- `scripts/evaluate_clip_flickr30k.py`: 加分项 CLIP 零样本检索评测脚本。
- `docs/开发记录/`: 阶段记录、问题和结论。
- `results/`: 实验指标输出。
- `submission/`: 最终打包文件输出目录。

## 环境

本机 `/mnt/kxh/miniconda3/envs/trl` 环境可直接用于加分项：

```bash
/mnt/kxh/miniconda3/envs/trl/bin/python - <<'PY'
import torch
import transformers
import datasets
print(torch.__version__, torch.cuda.is_available())
print(transformers.__version__, datasets.__version__)
PY
```

当前已验证版本：

- Python 3.10
- PyTorch 2.8.0 + CUDA 12.8
- Transformers 4.57.6
- Datasets 4.4.2
- 8 张 NVIDIA A100-SXM4-80GB 可见

VSRN 官方代码依赖 Python 2.7、PyTorch 0.4.1、NLTK、pycocotools 等老环境。为了忠实复现，
建议单独按官方 README 建一个旧环境运行 VSRN，不把依赖强行装进 `trl` 环境。

## 数据下载

### Flickr30k 原始图片与标注

标注来自 Hugging Face：

```bash
/mnt/kxh/miniconda3/envs/trl/bin/python - <<'PY'
from huggingface_hub import hf_hub_download

base = "/mnt/kxh/smx/homework/h2/datasets/flickr30k_hf"
for name in ["README.md", "flickr30k.py", "flickr_annotations_30k.csv"]:
    print(hf_hub_download("nlphuji/flickr30k", repo_type="dataset", filename=name, local_dir=base))
PY
```

图片包使用 GitHub release 分片下载：

```bash
cd /mnt/kxh/smx/homework/h2/datasets/flickr30k_github_parts
curl -L -C - -o flickr30k_part00 https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part00
curl -L -C - -o flickr30k_part01 https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part01
curl -L -C - -o flickr30k_part02 https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part02
cat flickr30k_part00 flickr30k_part01 flickr30k_part02 > flickr30k.zip
unzip -q flickr30k.zip -d /mnt/kxh/smx/homework/h2/datasets/flickr30k_images
```

数据完整性检查：

```bash
cd /mnt/kxh/smx/homework/h2
/mnt/kxh/miniconda3/envs/trl/bin/python scripts/inspect_flickr30k.py \
  --annotations-csv datasets/flickr30k_hf/flickr_annotations_30k.csv \
  --image-root datasets/flickr30k_images
```

### VSRN/SCAN 预计算特征

VSRN 官方训练命令需要 SCAN 提供的 bottom-up region features：

```bash
cd /mnt/kxh/smx/homework/h2/datasets/scan
wget -c https://scanproject.blob.core.windows.net/scan-data/data.zip
unzip data.zip
```

本机实测该 Azure Blob 链接经当前代理出现 TLS EOF；若网络恢复，可继续使用上述命令。
Kaggle 上也有同源 SCAN Faster R-CNN Image Features 数据集，但当前机器没有 Kaggle CLI/token。

## 基础项：VSRN 复现命令

VSRN 官方 Flickr30k 训练命令如下，`$DATA_PATH` 指向解压后的 SCAN `data/` 目录：

```bash
cd /mnt/kxh/smx/homework/h2/VSRN
python train.py \
  --data_path "$DATA_PATH" \
  --data_name f30k_precomp \
  --logger_name runs/flickr_VSRN \
  --max_violation \
  --lr_update 10 \
  --max_len 60
```

评测命令见 `VSRN/README.md` 的 `evaluation.evalrank(...)` 示例。

## 加分项：CLIP 零样本图文检索

使用较新的 CLIP 模型在 Flickr30k test split 上评测 Recall@K：

```bash
cd /mnt/kxh/smx/homework/h2
/mnt/kxh/miniconda3/envs/trl/bin/python scripts/evaluate_clip_flickr30k.py \
  --annotations-csv datasets/flickr30k_hf/flickr_annotations_30k.csv \
  --image-root datasets/flickr30k_images \
  --split test \
  --model-name openai/clip-vit-base-patch32 \
  --batch-size 128 \
  --output-json results/clip_flickr30k_test_metrics.json
```

先跑小样本验证：

```bash
cd /mnt/kxh/smx/homework/h2
/mnt/kxh/miniconda3/envs/trl/bin/python scripts/evaluate_clip_flickr30k.py \
  --annotations-csv datasets/flickr30k_hf/flickr_annotations_30k.csv \
  --image-root datasets/flickr30k_images \
  --split test \
  --max-images 32 \
  --output-json results/clip_flickr30k_smoke_metrics.json
```

输出 JSON 中包含：

- `image_to_text_r1/r5/r10`
- `text_to_image_r1/r5/r10`
- `rsum`

当前全量 test split 已完成一次评测，结果为：

| 模型 | Image-to-Text R@1 | R@5 | R@10 | Text-to-Image R@1 | R@5 | R@10 | RSum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `openai/clip-vit-base-patch32` | 79.40 | 95.00 | 98.10 | 58.84 | 83.46 | 90.04 | 504.84 |

## 提交打包

按课程要求，最终需要改名为：

- `姓名-学号-作业二报告.pdf`
- `姓名-学号-作业二代码.zip`

代码包建议排除数据、模型权重和中间产物：

```bash
cd /mnt/kxh/smx/homework/h2
zip -r submission/姓名-学号-作业二代码.zip \
  README.md scripts docs VSRN \
  -x '*/.git/*' '*/__pycache__/*'
```
