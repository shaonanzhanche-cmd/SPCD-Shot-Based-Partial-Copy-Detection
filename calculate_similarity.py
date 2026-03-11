import torch
import numpy as np
from util import *
from preprocess_videos import *
from label_transfer import seconds_to_time_str
from audio_extractor import convert_timestamps, extract_audio_features


def cosine_similarity(feature1, feature2):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
    similarity = cos(feature1, feature2)
    return similarity.item()


def calculate_similarity_matrix(video_a_features, video_b_features, pb_list, black_filter=False):
    num_shots_a = len(video_a_features)
    num_shots_b = len(video_b_features)
    similarity_matrix = np.zeros((num_shots_a, num_shots_b))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(num_shots_a):
        if black_filter and pb_list[i] >= 0.9:
            print("黑屏镜头跳过匹配！")
            continue
        for j in range(num_shots_b):
            similarity_matrix[i, j] = cosine_similarity(torch.tensor(video_a_features[i]).to(device),
                                                        torch.tensor(video_b_features[j]).to(device))
    return similarity_matrix


def get_time_range(timestamps, index):
    if index == 0:
        start_time = 0
    else:
        start_time = timestamps[index - 1]
    end_time = timestamps[index]
    time_range = str(start_time) + "--" + str(end_time)

    return time_range


# 单视频检索
def retrieval_by_similarity(features_data, video_to_retrieve, weights, threshold=0.9, black_filter=False):
    model_frame = FrameFeatureExtractor()
    model_shot = model_loader(weights)
    # model_shot = ShotFeatureExtractor()
    a_shot_boundaries, a_timestamps, a_shots_features = preprocess_a_video(video_to_retrieve, model_frame, model_shot)

    video_a_id = video_to_retrieve.split('\\')[-1]
    similarity_result = []

    if black_filter:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        black_screen_torch = torch.tensor(black_screen_feature).to(device)
        pb_list = [cosine_similarity(torch.tensor(shot).to(device), black_screen_torch) for shot in a_shots_features]
    else:
        pb_list = [None] * len(a_shots_features)

    for video_b_id, b_data in features_data.items():
        similarity_matrix = calculate_similarity_matrix(a_shots_features, b_data['shot_features'], pb_list, black_filter=black_filter)
        rows, cols = similarity_matrix.shape

        # 遍历所有可能的斜线，行列差是斜线的编号
        for diff in range(-(rows - 1), cols):
            start_row = max(0, -diff)
            start_col = max(0, diff)

            # 初始化合并区间
            last_start_a, last_start_b, last_end_a, last_end_b = None, None, None, None
            similarity_sums, nums = 0, 0
            # 沿斜线遍历
            row, col = start_row, start_col
            while row < rows and col < cols:
                similarity = similarity_matrix[row, col]

                if similarity > threshold:
                    # 当前相似片段
                    start_a, end_a = get_time_range(a_timestamps, row).split('--')
                    start_b, end_b = get_time_range(b_data['shot_timestamps'], col).split('--')
                    if last_start_a is not None and start_a == last_end_a and start_b == last_end_b:
                        # 更新合并区间
                        last_end_a = end_a
                        last_end_b = end_b
                        similarity_sums += similarity
                        nums += 1
                    else:
                        # 存储上一个合并区间
                        if last_start_a is not None:
                            similarity_result.append(
                                f"{video_a_id}\t{last_start_a}--{last_end_a}\t{video_b_id}\t{last_start_b}--{last_end_b}\t{similarity_sums / nums}")

                        # 开始新的区间
                        last_start_a, last_end_a = start_a, end_a
                        last_start_b, last_end_b = start_b, end_b
                        similarity_sums = similarity
                        nums = 1
                row += 1
                col += 1

            # 处理最后一条合并结果
            if last_start_a is not None:
                similarity_result.append(
                    f"{video_a_id}\t{last_start_a}--{last_end_a}\t{video_b_id}\t{last_start_b}--{last_end_b}\t{similarity_sums / nums}")

    return similarity_result


# 结果重排序
def sort_similarity_result(similarity_result):
    sorted_result = []

    # 首先按 B 视频分组，将每个 B 视频的结果分开
    from collections import defaultdict
    b_video_groups = defaultdict(list)

    for result in similarity_result:
        parts = result.split('\t')
        video_a_id = parts[0]
        a_start, a_end = map(float, parts[1].split('--'))  # 获取 A 视频的起始和结束时间
        video_b_id = parts[2]
        b_start, b_end = map(float, parts[3].split('--'))  # 获取 B 视频的起始和结束时间
        similarity = parts[4]

        # 将结果存储到对应的 B 视频组
        b_video_groups[video_b_id].append((video_a_id, a_start, a_end, video_b_id, b_start, b_end, similarity))

    # 对每个 B 视频组内的结果按 A 视频时间进行排序
    for video_b_id, group in b_video_groups.items():
        # 对 A 视频的起始时间排序，如果起始时间相同则比较结束时间
        sorted_group = sorted(group, key=lambda x: (x[1], x[2]))

        # 将排序后的结果添加到最终结果中
        for item in sorted_group:
            video_a_id, a_start, a_end, video_b_id, b_start, b_end, similarity = item

            # 将秒数转换为时分秒格式
            a_start_hms = seconds_to_time_str(a_start)
            a_end_hms = seconds_to_time_str(a_end)
            b_start_hms = seconds_to_time_str(b_start)
            b_end_hms = seconds_to_time_str(b_end)

            # 按照待检视频相似片段的始末时间先后重排序
            formatted_result = f"{video_a_id}\t{a_start_hms}--{a_end_hms}\t{video_b_id}\t{b_start_hms}--{b_end_hms}\t{similarity}"
            sorted_result.append(formatted_result)

    return sorted_result


# 批量搜索
def muti_retrievel(root_path, features_data, weights, result_path, threshold=0.9, black_filter=False):
    video_files = video_loader(root_path)
    for video_file in tqdm(video_files, desc='retrieving videos'):
        similarity_result = retrieval_by_similarity(features_data, video_file, weights, threshold=threshold,
                                                    black_filter=black_filter)
        sorted_result = sort_similarity_result(similarity_result)
        save_path = result_path + video_file.split('\\')[-2] + '.txt'
        with open(save_path, 'a') as file:
            for item in sorted_result:
                file.write(str(item) + '\n')


if __name__ == "__main__":
    feature_path = "projected_features/distraction-1000c"
    video_to_retrieve = "D:\\python workplace\\NDVR\\vcdb-core\\core_dataset\\beautiful_mind_game_theory\\46f2e964ae16f5c27fad70d6849c76616fad7502.flv"
    root_path = "D:\\python workplace\\NDVR\\vcdb-core-test\\core_dataset"
    result_path = "prediction_results/distraction3-black_filter/"
    weights = 'models/k=1.0/trained_model_50_0.9_0.5.pth'
    features_data = load_features(feature_path)
    # similarity_result = retrieval_by_similarity(features_data, video_to_retrieve, weights, threshold=0.9)
    # sorted_result = sort_similarity_result(similarity_result)
    # with open(result_path, 'w') as file:
    #     for item in sorted_result:
    #         file.write(str(item) + '\n')
    model_frame = FrameFeatureExtractor()
    model_shot = model_loader(weights)
    _, _, black_screen_features = preprocess_a_video("./tools/black_screen_demo.mp4", model_frame, model_shot)
    global black_screen_feature
    black_screen_feature = black_screen_features[0]
    muti_retrievel(root_path, features_data, weights, result_path, black_filter=True)
