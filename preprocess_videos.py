import cv2
import numpy as np
import torch
from util import *
from tqdm import tqdm
import torch.quantization
import time


def preprocess_video(video_path, model_frame, model_shot, regular_interval=10, min_len=1, max_len=60, window_size=10, k=1.5):
    # 加载预训练模型
    model_frame.eval()  # 设置为评估模式
    model_shot.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_frame.to(device)
    model_shot.to(device)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频每秒帧数
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数
    video_name = video_path.split('\\')[-1]
    print(f"video_name:{video_name}, FPS: {fps}, Total frames: {frame_count}")

    shot_boundaries = []
    timestamps = []
    shots_features = []
    current_shot_frames = []
    similarities_window = []

    previous_feature = None
    current_frame_number = 0

    while True:
        ret, frame = cap.read()

        # 为最后一个镜头提取特征
        if not ret:
            if len(current_shot_frames)/fps >= min_len:
                current_timestamp = round(current_frame_number / fps)
                shot_boundaries.append(current_frame_number)
                timestamps.append(current_timestamp)
                preprocessed_shot = preprocess_shot(current_shot_frames)
                with torch.no_grad():
                    shot_feature = model_shot(preprocessed_shot)
                shots_features.append(shot_feature)
            break

        if frame is None or frame.size == 0:
            print(f"Warning: Empty frame at frame {current_frame_number}")
            current_frame_number += 1
            continue

        current_shot_frames.append(frame)

        # 为超长镜头自动分割
        if len(current_shot_frames)/fps >= max_len:
            current_timestamp = round(current_frame_number / fps)
            shot_boundaries.append(current_frame_number)
            timestamps.append(current_timestamp)
            preprocessed_shot = preprocess_shot(current_shot_frames)
            with torch.no_grad():
                shot_feature = model_shot(preprocessed_shot)
            shots_features.append(shot_feature)
            current_shot_frames = []

        # 每隔regular_interval进行一次边界检测
        if current_frame_number % regular_interval == 0:
            # 预处理图像并提取特征
            input_tensor = preprocess_frame(frame).unsqueeze(0)  # 添加batch维度
            with torch.no_grad():
                current_feature = model_frame(input_tensor)

            # 如果这不是第一帧，计算与前一特征帧的相似度
            if previous_feature is not None:
                current_timestamp = round(current_frame_number / fps)
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                similarity = cos(previous_feature, current_feature)
                similarities_window.append(similarity.item())

                # 保证镜头长度在适当大小
                if min_len <= len(current_shot_frames)/fps < max_len:
                    # 如果窗口未满，使用当前数据计算动态阈值
                    current_window = similarities_window[-min(len(similarities_window), window_size):]
                    mean_sim = np.mean(current_window)  # 计算均值
                    std_sim = np.std(current_window)  # 计算标准差

                    # 动态阈值：均值 - k * 标准差
                    dynamic_threshold = mean_sim - k * std_sim
                    print(f"Frame: {current_frame_number}, Time: {current_timestamp}s, Similarity: {similarity.item()}, Threshold: {dynamic_threshold}")

                    # 基于相似度和其他逻辑识别镜头边界（这里可以调整条件）
                    if similarity.item() < dynamic_threshold:
                        shot_boundaries.append(current_frame_number)
                        timestamps.append(current_timestamp)
                        preprocessed_shot = preprocess_shot(current_shot_frames)
                        with torch.no_grad():
                            shot_feature = model_shot(preprocessed_shot)

                        shots_features.append(shot_feature)
                        current_shot_frames = []

            previous_feature = current_feature

        current_frame_number += 1

    # print(f"Successfully processed {video_name}.")
    cap.release()
    return shot_boundaries, timestamps, shots_features


def preprocess(root_path, output_path, model_frame, model_shot):
    video_files = video_loader(root_path)
    for video_file in tqdm(video_files, desc="Processing videos"):
        print("\n")
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            continue
        if check_exist(video_file.split('\\')[-1], output_path):
            continue
        shot_boundaries, timestamps, shots_features = preprocess_a_video(video_file, model_frame, model_shot)
        update_features(video_file.split('\\')[-1], timestamps, shots_features, output_path)
    print("All videos have been successfully processed.")


def preprocess_a_video(video_path, model_frame, model_shot):
    print("--------------------------------------------------------------")
    print("Video ID:", video_path.split('\\')[-1])
    shot_boundaries, timestamps, shots_features = preprocess_video(video_path, model_frame, model_shot)
    print("Suspected shot boundaries (Shot boundaries):", shot_boundaries)
    print("Corresponding timestamps (seconds):", timestamps)
    print("Shot features shape:", len(shots_features))
    return shot_boundaries, timestamps, shots_features


def print_features(feature_path):
    features_data = load_features(feature_path)
    i = 1
    for video_id, data in features_data.items():
        print(f"Video number: {i}")
        print(f"Video ID: {video_id}")
        print(f"Shot Timestamps: {data['shot_timestamps']}")
        print(f"Shot Features Length: {len(data['shot_features'])}")
        i += 1


if __name__ == "__main__":
    root_path = "D:\\python workplace\\NDVR\\vcdb-diy\\vcdb-diy-query\\videos"
    output_path = 'process_time/diy-1'
    weights = 'models/k=1.0/trained_model_50_0.9_0.5.pth'
    model_frame = FrameFeatureExtractor()
    model_shot = model_loader(weights)
    # preprocess(root_path, output_path, model_frame, model_shot)
    # print_features(output_path)
    preprocess_a_video("./tools/black_screen_demo.mp4", model_frame, model_shot)
