# SPCD — Shot-Based Partial Copy Detection

[中文](#中文说明) | [English](#english)

A shot-based partial video copy detection pipeline built around **inter-shot similarity measurement**.

- Shot segmentation via a **dynamic threshold** on frame-to-frame cosine similarity
- Shot-level feature extraction + (optional) learned **feature projection**
- Retrieval by scanning **diagonals of the shot similarity matrix** to merge consecutive matching shots
- Utilities for result sorting/filtering and offline evaluation (precision/recall)

> Note: This repository appears to be a research/experimental codebase. Some scripts contain hard-coded local paths (e.g. `D:\\python workplace\\...`). You will likely need to adapt paths to your environment.

---

## 中文说明

### 1. 项目简介

本项目实现了一种**基于镜头（shot）**的局部视频拷贝检测（Partial Copy Detection, PCD）方法，通过计算**跨视频镜头级特征的相似度**来定位相似片段。整体流程包括：

1) 视频镜头切分（shot segmentation）
2) 镜头级特征提取（shot feature extraction）
3)（可选）特征投影/度量学习（feature projection）
4) 基于镜头相似度矩阵的检索（retrieval）与结果后处理
5) 结果评估（precision / recall）

### 2. 代码结构（主要文件）

- `preprocess_videos.py`：对单个视频进行镜头切分并提取镜头特征
- `calculate_similarity.py`：
  - 计算镜头相似度矩阵（cosine similarity）
  - 通过遍历相似度矩阵的斜线（diagonal）合并连续相似镜头，输出候选片段
  - 支持可选的黑屏镜头过滤（black_filter）
- `trainer.py`：训练特征投影网络（FeatureProjector）并保存权重；也包含对预处理特征进行投影的函数
- `result_filter.py`：对预测结果做包含关系过滤，保留覆盖范围更大的片段
- `results_analyse.py`：读取预测与标注结果，计算 Precision/Recall
- `model.py` / `util.py` / `dataset_constructor.py`：模型与数据集构建、工具函数
- `datasets/*.pkl`：示例/中间数据（成对训练数据集）
- `models/**/*.pth`：示例模型权重
- `prediction_results/`：示例预测结果输出
- `tools/`：可视化/辅助脚本与演示媒体

### 3. 环境依赖

代码主要基于 Python + PyTorch + OpenCV。

本仓库已提供 `requirements.txt`，可直接安装：

```bash
pip install -r requirements.txt
```

说明：
- `requirements.txt` 同时包含“核心依赖”和 `tools/`、`audio_extractor.py` 用到的**可选依赖**；若你只跑核心流程（镜头切分/特征/检索），可不使用音频与可视化相关包。
- PyTorch/torchvision 建议按你的 CUDA 版本在本机选择合适的安装方式。

### 4. 快速开始（端到端流程）

下面给出一个“典型工作流”的最小说明（你需要把路径改成自己的数据路径）。

#### 4.1 准备镜头特征库（gallery / reference）

思路：对库视频逐个做镜头切分与镜头特征提取，并将特征保存到某个目录（例如 `preprocessed_features/...`）。

入口参考：`preprocess_videos.py:148` 附近提供了示例调用方式。

#### 4.2（可选）训练特征投影模型

- 训练入口：`trainer.py:47`（`main()`）
- 数据集示例：`datasets/paired_dataset_k=1.0.pkl` 等
- 训练输出：`models/.../*.pth`

示例（按你的实际路径调整）：

```bash
python trainer.py
```

#### 4.3（可选）对已提取的镜头特征做投影

- 入口：`trainer.py:85`（`feature_projection()`）

投影后的特征可用于后续检索（例如 `projected_features/...`）。

#### 4.4 检索（对 query 视频进行局部拷贝检测）

- 核心入口：`calculate_similarity.py:41`（`retrieval_by_similarity()` / `muti_retrievel()`）
- 输出：`prediction_results/.../*.txt`

脚本底部 `calculate_similarity.py:161` 给出了示例参数（包含黑屏过滤示例）。

#### 4.5 结果过滤（去掉被包含的片段）

- 入口：`result_filter.py:47`

```bash
python result_filter.py
```

#### 4.6 评估（Precision / Recall）

- 入口：`results_analyse.py:128`
- 需要准备 ground truth 标注文件夹（脚本中当前是硬编码路径，需要改成你自己的）

```bash
python results_analyse.py
```

### 5. 输出格式（预测结果）

预测结果文本通常按行存储，字段用 `\t` 分隔，形式为：

```
<videoA>\t<startA>--<endA>\t<videoB>\t<startB>--<endB>\t<confidence>
```

- `start/end`：在不同阶段可能是秒数或 `hh:mm:ss`（见 `sort_similarity_result()` 中的转换）
- `confidence`：相似度均值（合并的镜头段落内）


## English

### 1. Overview

This repository implements a **shot-based partial video copy detection** approach via **inter-shot similarity measurement**. A typical pipeline is:

1) Shot segmentation
2) Shot-level feature extraction
3) (Optional) feature projection / metric learning
4) Retrieval by analyzing the shot similarity matrix and merging consecutive matches
5) Post-processing and evaluation

### 2. Key files

- `preprocess_videos.py`: shot segmentation + shot feature extraction for a video
- `calculate_similarity.py`:
  - builds a shot-to-shot similarity matrix (cosine similarity)
  - scans diagonals to merge consecutive matching shots into segments
  - optional black-screen shot filtering (`black_filter`)
- `trainer.py`: trains a `FeatureProjector` and saves `.pth` weights; also includes feature projection for precomputed features
- `result_filter.py`: removes contained segments, keeping larger ranges
- `results_analyse.py`: parses predictions/ground truth and computes precision/recall
- `datasets/*.pkl`: paired datasets for training (examples)
- `models/**/*.pth`: example weights
- `prediction_results/`: example outputs
- `tools/`: helper scripts and demo media

### 3. Dependencies

The code is mainly built on Python + PyTorch + OpenCV.

This repo provides a `requirements.txt`:

```bash
pip install -r requirements.txt
```

Notes:
- `requirements.txt` includes both core dependencies and **optional** packages used by `tools/` and `audio_extractor.py`.
- Install PyTorch/torchvision in a way that matches your local CUDA setup.

### 4. Quickstart (typical workflow)

You will need to edit paths to match your environment.

1) Extract shot features for your reference/gallery videos (`preprocess_videos.py`)
2) (Optional) Train a projector (`trainer.py`) using `datasets/*.pkl`
3) (Optional) Project existing features (`trainer.py:feature_projection`)
4) Retrieve matches for query videos (`calculate_similarity.py`)
5) Filter redundant contained segments (`result_filter.py`)
6) Evaluate precision/recall (`results_analyse.py`)

### 5. Prediction output format

Each line typically looks like:

```
<videoA>\t<startA>--<endA>\t<videoB>\t<startB>--<endB>\t<confidence>
```
