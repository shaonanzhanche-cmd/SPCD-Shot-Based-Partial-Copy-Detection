import torch
import librosa
import numpy as np
from util import *
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from torchvggish import vggish
from torchvggish.vggish_input import waveform_to_examples


# 1. 从视频中根据时间戳提取音频并保存为临时文件
def extract_audio(video_path, timestamps, audio_temp_paths=None):
    if audio_temp_paths is None:
        audio_temp_paths = [f"temp_audio_{i}.wav" for i in range(len(timestamps))]

    video = VideoFileClip(video_path)
    if not video.audio:
        print(f"警告：视频 {video_path} 没有音频轨道，将使用空音频处理。")
        # 为每个时间戳创建一个空的临时文件路径
        for i in range(len(timestamps)):
            open(audio_temp_paths[i], 'a').close()
        return audio_temp_paths

    video_duration = video.duration
    valid_timestamps = []
    for i, (start, end) in enumerate(timestamps):
        # 处理最后一个时间戳
        if i == len(timestamps) - 1 and end > video_duration:
            end = video_duration
        if start < end and start < video_duration:
            valid_timestamps.append((start, end))
        else:
            print(f"忽略时间戳 ({start}, {end})，时间范围无效。")

    for i, (start, end) in enumerate(valid_timestamps):
        # 截断音频
        audio_clip = video.audio.subclip(start, end)
        # 导出音频为临时文件（16kHz单声道）
        audio_clip.write_audiofile(audio_temp_paths[i], fps=16000, codec='pcm_s16le')

    return audio_temp_paths


# 2. 预处理音频生成VGGish输入
def audio_to_vggish_input(audio_path):
    try:
        # 检查文件大小是否为 0（空文件）
        if os.path.getsize(audio_path) == 0:
            print(f"警告：音频文件 {audio_path} 为空，将返回空张量。")
            return torch.tensor([])
        # 加载音频波形（自动重采样到16kHz）
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_duration = len(waveform) / sr  # 计算音频时长
        min_duration = 0.01  # 设置最小时长阈值
        if audio_duration < min_duration:
            print(f"警告：音频 {audio_path} 时长 {audio_duration} 秒，过短，可能会导致异常输出。")

        # 转换为VGGish输入格式（直接使用官方预处理函数）
        input_features = waveform_to_examples(waveform, sr)
        print(f"音频 {audio_path} 转换后的输入特征形状: {input_features.shape}")

        # 转换为PyTorch张量并添加批次维度
        input_tensor = torch.tensor(input_features)  # [N, 1, 96, 64]
        return input_tensor
    except Exception as e:
        print(f"处理音频 {audio_path} 时出错: {e}，返回空张量。")
        return torch.tensor([])


# 3. 加载预训练的VGGish模型
def load_vggish_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = vggish()
    # model.load_state_dict(torch.load('pytorch_vggish/vggish-10086976.pth'))
    model = model.to(device).eval()
    if hasattr(model.pproc, '_pca_matrix'):
        model.pproc._pca_matrix = model.pproc._pca_matrix.to(device)
    if hasattr(model.pproc, '_pca_means'):
        model.pproc._pca_means = model.pproc._pca_means.to(device)
    return model


# 主流程
def extract_audio_features(video_path, timestamps):
    # 提取音频
    audio_paths = extract_audio(video_path, timestamps)

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_vggish_model(device)

    all_embeddings = []
    for audio_path in audio_paths:
        # 预处理音频
        input_tensor = audio_to_vggish_input(audio_path)

        if input_tensor.numel() == 0:
            # 如果输入张量为空，生成零向量作为特征
            embeddings = torch.zeros(1, 128)
        else:
            # 推理
            with torch.no_grad():
                embeddings = model(input_tensor.to(device))

            # 压缩特征到 (1, 128)
            if len(embeddings.shape) == 1:
                embeddings = embeddings.unsqueeze(0)
            elif embeddings.shape[0] > 1:
                embeddings = embeddings.mean(dim=0, keepdim=True)

        all_embeddings.append(embeddings.cpu().numpy())

    # 删除临时的wav文件
    for audio_path in audio_paths:
        try:
            os.remove(audio_path)
            print(f"已删除临时文件: {audio_path}")
        except OSError as e:
            print(f"删除文件 {audio_path} 时出错: {e}")

    return all_embeddings


# 将单一时间戳列表转换为 (start, end) 元组列表
def convert_timestamps(timestamps):
    result = []
    start = 0
    for end in timestamps:
        result.append((start, end))
        start = end
    return result


def fuse_all_features(root_path, features_path, output_path):
    video_files = video_loader(root_path)
    features_data = load_features(features_path)

    for video_file in tqdm(video_files, desc="Processing videos"):
        print("\n")
        if check_exist(video_file.split('\\')[-1], output_path):
            continue
        video_id = video_file.split('\\')[-1]
        print("--------------------------------------------------------------")
        print("Video ID:", video_id)
        timestamps = features_data[video_id]['shot_timestamps']
        converted_timestamps = convert_timestamps(timestamps)
        shot_features = features_data[video_id]['shot_features']
        audio_features = extract_audio_features(video_file, converted_timestamps)
        fused_features = []
        for i in range(len(shot_features)):
            fused_feature = np.concatenate((audio_features[i], shot_features[i]), axis=1)
            fused_features.append(fused_feature)
        save_fused_features(video_file.split('\\')[-1], timestamps, fused_features, output_path)

    print("All videos have been successfully processed.")


# 使用示例
if __name__ == "__main__":
    # video_path = "D:\\python workplace\\NDVR\\vcdb-core\\core_dataset\\baggio_penalty_1994\\6d1a89c83d554fc6a5e39fcadb172a79baf140fd.mp4"  # 支持MP4/FLV等格式
    root_path = "D:\\python workplace\\NDVR\\vcdb-core\\core_dataset"
    features_path = 'preprocessed_features/shot_features_core_v2_k=1.5'
    output_path = 'preprocessed_features/fused_features_k=1.5'
    fuse_all_features(root_path, features_path, output_path)
