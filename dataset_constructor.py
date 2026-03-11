import pickle
import random
from util import load_features
import os
import re


def paired_dataset_constructor(features_data, annotations_path, save_path):
    # 存储相似镜头，方便后续随机生成不相似镜头
    similar_shots = []
    # 定义存储标注的列表
    positive_pairs = []
    negative_pairs = []

    # 正则表达式匹配多个相似对，例如 (0, 1)(1, 2)(2, 3)
    pair_pattern = r'\((\d+),\s*(\d+)\)'

    # 构造相似对
    for file_name in os.listdir(annotations_path):
        if file_name.endswith('.txt'):  # 只处理txt文件
            file_path = os.path.join(annotations_path, file_name)

            # 打开并读取txt文件
            with open(file_path, 'r') as file:
                for line in file:
                    # 移除首尾的空白字符（包括换行符）
                    line = line.strip()

                    if line:
                        # 检查是否为空行
                        if not line:
                            continue

                        # 检查是否有足够的值
                        if '\t' not in line:
                            print(f"Skipping malformed line: {line}")
                            continue

                        # 按照格式分割数据
                        video_files, index_tuples_str = line.split('\t')
                        videoA, videoB = video_files.split(',')

                        if videoA == videoB:
                            continue

                        # 使用正则表达式提取所有的索引对
                        matches = re.findall(pair_pattern, index_tuples_str)

                        # 遍历提取到的每个索引对
                        for match in matches:
                            indexA, indexB = map(int, match)  # 将字符串转换为整数

                            print(f"{videoA},{videoB},{indexA},{indexB}")
                            featureA = features_data[videoA]['shot_features'][indexA]
                            featureB = features_data[videoB]['shot_features'][indexB]
                            label = 1

                            positive_pairs.append((featureA, featureB, label))
                            similar_shots.append((videoA, videoB, indexA, indexB))

    # 构造不相似对，确保视频A和视频B不同
    all_video_names = list(features_data.keys())
    existing_pairs = set()  # 用于存储已经生成的不相似对

    for _ in range(len(positive_pairs)):  # 构造与相似对数量相等的不相似对
        while True:
            videoA, videoB = random.sample(all_video_names, 2)  # 随机选择两个不同的视频

            # 从视频A和视频B中随机选择两个不相似的片段
            indexA = random.randint(0, len(features_data[videoA]['shot_features']) - 1)
            indexB = random.randint(0, len(features_data[videoB]['shot_features']) - 1)

            featureA = features_data[videoA]['shot_features'][indexA]
            featureB = features_data[videoB]['shot_features'][indexB]

            # 确保它们不在相似对中
            if (videoA, videoB, indexA, indexB) not in similar_shots and \
                    (videoB, videoA, indexB, indexA) not in similar_shots and \
                    (videoA, videoB, indexA, indexB) not in existing_pairs and \
                    (videoB, videoA, indexB, indexA) not in existing_pairs:
                negative_pairs.append((featureA, featureB, 0))  # 不相似对标签为 0
                existing_pairs.add((videoA, videoB, indexA, indexB))  # 加入已生成对的集合
                break

    # 将正样本和负样本组合
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)  # 打乱数据集

    # 将数据集保存到文件
    with open(save_path, 'wb') as f:
        pickle.dump(all_pairs, f)


def load_dataset(dataset_path):
    # 从文件加载数据集
    with open(dataset_path, 'rb') as f:
        all_pairs = pickle.load(f)

    return all_pairs


if __name__ == "__main__":
    # 加载特征向量
    feature_path = "preprocessed_features/fused_features_k=1.0"
    features_data = load_features(feature_path)
    # 数据集保存路径
    dataset_path = 'datasets/paired_dataset_fused_k=1.0.pkl'
    # 遍历文件夹中的所有txt文件
    annotations_path = 'annotations_k=1.0'

    paired_dataset_constructor(features_data, annotations_path, dataset_path)
    # dataset = load_dataset(dataset_path)
    # # 访问加载的数据集
    # print(len(dataset))










