import pickle
import torch
import cv2
import torchvision.transforms as transforms
import os
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from model import *


def preprocess_shot(shot_frames, resize_shape=(112, 112), num_frames=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Compose([
        Resize(resize_shape),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    selected_frames = []
    total_frames = len(shot_frames)

    # 筛选参与特征计算的帧
    if total_frames >= num_frames:
        step = total_frames // num_frames
        selected_indices = [i * step for i in range(num_frames)]
    else:
        # 当帧数不足时，重复选取帧直到达到num_frames
        repeat_factor = num_frames // total_frames
        additional_frames = num_frames % total_frames
        selected_indices = list(range(total_frames)) * repeat_factor + list(range(additional_frames))

    for idx in selected_indices:
        frame = shot_frames[idx]
        frame_tensor = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).to(device)
        selected_frames.append(frame_tensor)

    # 将帧堆叠成一个4D张量，形状为[T, C, H, W]
    processed_frames = torch.stack(selected_frames).to(device)

    # 调整维度以匹配模型输入要求
    processed_frames = processed_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)  # 形状为[1, C, T, H, W]

    return processed_frames


def preprocess_frame(frame):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 将BGR转换为RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    # 定义预处理
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    processed_frame = preprocess(image).to(device)
    return processed_frame


def update_features(video_id, shot_timestamps, shot_features, output_path):
    all_videos_features = {}

    # Check if the pickle file already exists and load existing data
    if os.path.exists(output_path):
        with open(output_path, 'rb') as f:
            all_videos_features = pickle.load(f)

    # Update with new video features
    all_videos_features[video_id] = {
        'shot_timestamps': shot_timestamps,
        'shot_features': [feature.detach().cpu().numpy() for feature in shot_features]
    }

    # Write back to pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(all_videos_features, f)


def save_fused_features(video_id, shot_timestamps, shot_features, output_path):
    all_videos_features = {}

    # Check if the pickle file already exists and load existing data
    if os.path.exists(output_path):
        with open(output_path, 'rb') as f:
            all_videos_features = pickle.load(f)

    # Update with new video features
    all_videos_features[video_id] = {
        'shot_timestamps': shot_timestamps,
        'shot_features': shot_features
    }

    # Write back to pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(all_videos_features, f)


def check_exist(video_id, output_path):
    if os.path.exists(output_path):
        with open(output_path, 'rb') as f:
            all_videos_features = pickle.load(f)
    else:
        all_videos_features = {}
    if video_id in all_videos_features:
        print(f"Video {video_id} has already been processed. Skipping...")
        return True


def load_features(input_path):
    # Check if the pickle file exists
    if not os.path.exists(input_path):
        print(f"No such file: {input_path}")
        return {}

    # Load and return the data
    with open(input_path, 'rb') as f:
        all_videos_features = pickle.load(f)

    return all_videos_features


def video_loader(root_path):
    video_files = []
    video_extensions = ('.mp4', '.flv', '.mov')
    # Walk through all directories and files in root_folder
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Filter and add files with the specified extensions
        for filename in filenames:
            if filename.lower().endswith(video_extensions):
                video_files.append(os.path.join(dirpath, filename))

    return video_files


def annotation_loader(root_path):
    annotation_files = []
    annotation_extensions = '.txt'
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.lower().endswith(annotation_extensions):
                annotation_files.append(os.path.join(dirpath, filename))

    return annotation_files


def model_loader(weights_path, input_dim=512):
    feature_extractor = ShotFeatureExtractor()
    feature_projector = FeatureProjector(input_dim=input_dim)
    feature_projector.load_state_dict(torch.load(weights_path))  # 加载之前训练好的权重
    feature_projector.eval()  # 设置为评估模式，不会进行参数更新
    model_shot = EndToEndModel(feature_extractor, feature_projector)
    return model_shot


# def model_loader_int8(weights_path, input_dim=512):
#     feature_extractor = ShotFeatureExtractor()
#     feature_projector = FeatureProjector(input_dim=input_dim)
#     feature_projector.load_state_dict(torch.load(weights_path))  # 加载之前训练好的权重
#     feature_projector.eval()  # 设置为评估模式，不会进行参数更新
#     quantized_feature_extractor = torch.quantization.quantize_dynamic(
#         feature_extractor, {nn.Linear, nn.Conv3d}, dtype=torch.qint8
#     )
#     quantized_feature_projector = torch.quantization.quantize_dynamic(
#         feature_projector, {nn.Linear}, dtype=torch.qint8
#     )
#     model_shot = EndToEndModel(quantized_feature_extractor, quantized_feature_projector)
#     return model_shot
