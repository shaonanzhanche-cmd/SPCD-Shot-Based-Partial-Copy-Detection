# import os
# import random
# from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
#
#
# def sec_to_time(seconds):
#     """将秒转换为 00:00:00 时分秒格式"""
#     m, s = divmod(seconds, 60)
#     h, m = divmod(m, 60)
#     return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
#
#
# def temporal_augmentation(clip, target_speed=1.0):
#     """时间增强：调整视频速度（忽略音频）"""
#     return clip.speedx(target_speed).without_audio()
#
#
# def spatial_augmentation(clip, target_size=None, target_fps=24):
#     """
#     空间增强：随机选择一种方法进行处理：
#         - 'rotate'：随机旋转 [-10, 10] 度（bicubic 重采样）
#         - 'brightness'：随机调整亮度
#         - 'mirror'：水平镜像翻转
#     同时统一尺寸与帧率
#     """
#     methods = ['rotate', 'brightness', 'mirror']
#     chosen = random.choice(methods)
#     if chosen == 'rotate':
#         angle = random.randint(-10, 10)
#         clip = clip.fx(vfx.rotate, angle=angle, resample='bicubic')
#     elif chosen == 'brightness':
#         clip = clip.fx(vfx.lum_contrast, lum=random.uniform(0.8, 1.2))
#     elif chosen == 'mirror':
#         clip = clip.fl_image(lambda img: img[:, ::-1])
#     if target_size:
#         clip = clip.resize(target_size)
#     clip = clip.set_fps(target_fps)
#     return clip
#
#
# def generate_dataset(source_video_path, target_videos_dir, output_base_dir):
#     annotation_lines = []
#
#     # 打开源视频（在循环外打开，避免多次加载）
#     source_clip = VideoFileClip(source_video_path).without_audio()
#
#     # 获取目标视频文件路径列表
#     target_videos_paths = [
#         os.path.join(target_videos_dir, fname)
#         for fname in os.listdir(target_videos_dir)
#         if fname.lower().endswith(('.mp4', '.flv', '.avi'))
#     ]
#
#     target_fps = 24  # 统一帧率
#
#     for idx, target_path in enumerate(target_videos_paths[:10]):
#         target_video_name = os.path.basename(target_path)
#
#         # 每次从源视频中随机截取一个片段，并确保不超过原视频时长
#         min_duration = 1.0  # 最小持续时间（秒）
#         if source_clip.duration <= min_duration:
#             start_time = 0
#             duration = source_clip.duration
#         else:
#             start_time = random.uniform(0, source_clip.duration - min_duration)
#             max_possible_duration = min(60, source_clip.duration - start_time)
#             duration = random.uniform(min_duration, max_possible_duration)
#         end_time = start_time + duration
#         raw_query_clip = source_clip.subclip(start_time, end_time)
#
#         # 随机时间增强
#         new_query_clip = temporal_augmentation(
#             raw_query_clip, target_speed=random.choice([0.8, 1.2])
#         )
#
#         # 打开目标视频，获取目标尺寸
#         with VideoFileClip(target_path).without_audio() as target_clip:
#             target_clip = target_clip.set_fps(target_fps)
#             query_size = target_clip.size  # 以目标视频尺寸作为统一尺寸
#
#             # 随机选择一种空间增强方法（旋转、亮度调整或镜像）
#             new_query_clip = spatial_augmentation(new_query_clip, target_size=query_size, target_fps=target_fps)
#
#             # 预渲染变换后的查询片段，确保后续拼接稳定
#             temp_query_path = os.path.join(output_base_dir, f"temp_query_{idx}.mp4")
#             new_query_clip.write_videofile(
#                 temp_query_path,
#                 codec='libx264',
#                 fps=target_fps,
#                 bitrate="5000k",
#                 audio=False
#             )
#             new_query_clip = VideoFileClip(temp_query_path).without_audio()
#
#             # 随机选择插入位置
#             insert_pos = random.choice(['start', 'middle', 'end'])
#             if insert_pos == 'start':
#                 insert_time = 0
#             elif insert_pos == 'middle':
#                 insert_time = target_clip.duration / 2
#             else:
#                 insert_time = max(0, target_clip.duration - new_query_clip.duration)
#
#             # 分段拼接
#             segments = []
#             if insert_time > 0:
#                 segments.append(target_clip.subclip(0, insert_time))
#             segments.append(new_query_clip)
#             if insert_time < target_clip.duration:
#                 segments.append(target_clip.subclip(insert_time))
#
#             final_clip = concatenate_videoclips(segments, method="compose")
#
#             # 输出文件名与目标视频名保持一致
#             output_path = os.path.join(output_base_dir, target_video_name)
#             final_clip.write_videofile(
#                 output_path,
#                 codec='libx264',
#                 fps=target_fps,
#                 bitrate="5000k",
#                 audio=False
#             )
#
#             # 生成标注信息
#             source_start = sec_to_time(start_time)
#             source_end = sec_to_time(end_time)
#             target_start = sec_to_time(insert_time)
#             target_end = sec_to_time(insert_time + new_query_clip.duration)
#             source_name = os.path.basename(source_video_path)
#             line = f"{source_name},{target_video_name},{source_start},{source_end},{target_start},{target_end}\n"
#             annotation_lines.append(line)
#
#             # 删除预渲染的临时查询片段文件
#             if os.path.exists(temp_query_path):
#                 os.remove(temp_query_path)
#
#     # 关闭源视频
#     source_clip.close()
#
#     # 保存标注文件
#     annotation_path = os.path.join(output_base_dir, "annotation.txt")
#     with open(annotation_path, "w") as f:
#         f.writelines(annotation_lines)
#
#
# if __name__ == "__main__":
#     # 配置参数
#     SOURCE_VIDEO_PATH = r"D:\python workplace\NDVR\vcdb-diy-query\10\7eacnsjF-9s.mp4"  # 源视频路径
#     TARGET_VIDEOS_DIR = r"D:\python workplace\NDVR\vcdb-diy\10"  # 目标视频文件夹路径
#     OUTPUT_DIR = r"D:\python workplace\NDVR\vcdb-gen\10"  # 输出目录
#
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     generate_dataset(SOURCE_VIDEO_PATH, TARGET_VIDEOS_DIR, OUTPUT_DIR)
#     print("数据集生成完成！标注文件已保存。")


import os
import shutil
import random
import time
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx


def sec_to_time(seconds):
    """将秒转换为 00:00:00 时分秒格式"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


def temporal_augmentation(clip, target_speed=1.0):
    """时间增强：调整视频速度（忽略音频）"""
    return clip.speedx(target_speed).without_audio()


def spatial_augmentation(clip, target_size=None, target_fps=24):
    """
    空间增强：随机选择一种方法进行处理：
      - 'rotate'：随机旋转 [-10, 10] 度（bicubic 重采样）
      - 'brightness'：随机调整亮度
      - 'mirror'：水平镜像翻转
    同时统一尺寸与帧率
    """
    methods = ['rotate', 'brightness', 'mirror']
    chosen = random.choice(methods)
    if chosen == 'rotate':
        angle = random.randint(-10, 10)
        clip = clip.fx(vfx.rotate, angle=angle, resample='bicubic')
    elif chosen == 'brightness':
        clip = clip.fx(vfx.lum_contrast, lum=random.uniform(0.8, 1.2))
    elif chosen == 'mirror':
        clip = clip.fx(vfx.mirror_x)
    if target_size:
        clip = clip.resize(target_size)
    clip = clip.set_fps(target_fps)
    return clip


def safe_remove(filepath, retries=3, delay=2):
    """尝试删除文件，如果因占用而失败，等待重试几次"""
    for i in range(retries):
        try:
            os.remove(filepath)
            return
        except PermissionError:
            time.sleep(delay)
    print(f"WARNING: 无法删除临时文件 {filepath}")


def process_videos(query_base_dir, target_base_dir, output_base_dir, annotation_dir, num_folders=10):
    os.makedirs(annotation_dir, exist_ok=True)
    for folder_num in range(1, num_folders + 1):
        print(f"Processing group {folder_num} ...")
        # 构建对应的目录路径
        query_dir = os.path.join(query_base_dir, str(folder_num))
        target_dir = os.path.join(target_base_dir, str(folder_num))
        output_dir = os.path.join(output_base_dir, str(folder_num))
        os.makedirs(output_dir, exist_ok=True)

        # 标注文件保存路径
        annotation_path = os.path.join(annotation_dir, f"{folder_num}.txt")
        done_targets = set()
        # 如果标注文件已存在，则读取原有标注内容
        if os.path.exists(annotation_path):
            with open(annotation_path, "r", encoding="utf-8") as f:
                annotation_lines = f.readlines()
            for line in annotation_lines:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    done_targets.add(parts[1])
        else:
            annotation_lines = []

        # 获取查询视频文件路径列表（假设每个分组下仅有一个源视频；若有多个，则均处理）
        query_videos_paths = [
            os.path.join(query_dir, fname)
            for fname in os.listdir(query_dir)
            if fname.lower().endswith(('.mp4', '.flv', '.avi'))
        ]

        # 获取目标视频文件路径列表
        target_videos_paths = [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.lower().endswith(('.mp4', '.flv', '.avi'))
        ]

        target_fps = 24  # 统一帧率

        for query_path in query_videos_paths:
            query_video_name = os.path.basename(query_path)
            print(f"  Processing source video: {query_video_name}")

            # 复制原视频到输出目录作为完全重复视频
            output_query_path = os.path.join(output_dir, query_video_name)
            if not os.path.exists(output_query_path):
                shutil.copy(query_path, output_query_path)

            # 生成原视频的标注信息（完全重复视频）
            with VideoFileClip(query_path) as qc:
                full_duration = qc.duration
            source_start_full = "00:00:00"
            source_end_full = sec_to_time(full_duration)
            line_full = f"{query_video_name},{query_video_name},{source_start_full},{source_end_full},{source_start_full},{source_end_full}\n"
            # 如果未记录则添加
            if query_video_name not in done_targets:
                annotation_lines.append(line_full)
                done_targets.add(query_video_name)

            for target_path in target_videos_paths:
                target_video_name = os.path.basename(target_path)
                # 如果当前目标视频已处理，跳过（断点续传）
                if target_video_name in done_targets:
                    print(f"    Skip target video: {target_video_name} (already processed)")
                    continue
                print(f"    Inserting into target video: {target_video_name}")

                # 打开源视频，并随机截取一个片段（至少5秒）
                with VideoFileClip(query_path).without_audio() as source_clip:
                    min_duration = 5.0  # 最小持续时间（秒）
                    if source_clip.duration <= min_duration:
                        start_time = 0
                        duration = source_clip.duration
                    else:
                        start_time = random.uniform(0, source_clip.duration - min_duration)
                        max_possible_duration = min(60, source_clip.duration - start_time)
                        duration = random.uniform(min_duration, max_possible_duration)
                    end_time = start_time + duration
                    raw_query_clip = source_clip.subclip(start_time, end_time)

                    # 随机时间增强
                    new_query_clip = temporal_augmentation(
                        raw_query_clip, target_speed=random.choice([0.8, 1.2])
                    )

                # 打开目标视频，获取目标尺寸
                with VideoFileClip(target_path).without_audio() as target_clip:
                    target_clip = target_clip.set_fps(target_fps)
                    query_size = target_clip.size  # 以目标视频尺寸作为统一尺寸

                    # 随机选择一种空间增强方法
                    new_query_clip = spatial_augmentation(new_query_clip, target_size=query_size, target_fps=target_fps)

                    # 预渲染变换后的查询片段到临时文件
                    temp_query_filename = f"temp_query_{os.path.splitext(target_video_name)[0]}.mp4"
                    temp_query_path = os.path.join(output_dir, temp_query_filename)
                    new_query_clip.write_videofile(
                        temp_query_path,
                        codec='libx264',
                        fps=target_fps,
                        bitrate="5000k",
                        audio=False,
                        verbose=False,
                        logger=None
                    )
                    new_query_clip.close()  # 释放文件句柄
                    new_query_clip = VideoFileClip(temp_query_path).without_audio()

                    # 随机选择插入位置：开始、中间或末尾
                    insert_pos = random.choice(['start', 'middle', 'end'])
                    if insert_pos == 'start':
                        insert_time = 0
                    elif insert_pos == 'middle':
                        insert_time = target_clip.duration / 2
                    else:
                        insert_time = max(0, target_clip.duration - new_query_clip.duration)

                    # 分段拼接
                    segments = []
                    if insert_time > 0:
                        segments.append(target_clip.subclip(0, insert_time))
                    segments.append(new_query_clip)
                    if insert_time < target_clip.duration:
                        segments.append(target_clip.subclip(insert_time))

                    final_clip = concatenate_videoclips(segments, method="compose")

                    # 输出文件名与目标视频一致，保存到输出目录
                    output_path = os.path.join(output_dir, target_video_name)
                    final_clip.write_videofile(
                        output_path,
                        codec='libx264',
                        fps=target_fps,
                        bitrate="5000k",
                        audio=False,
                        verbose=False,
                        logger=None
                    )

                    # 生成插入标注信息
                    src_start_str = sec_to_time(start_time)
                    src_end_str = sec_to_time(end_time)
                    tgt_start_str = sec_to_time(insert_time)
                    tgt_end_str = sec_to_time(insert_time + new_query_clip.duration)
                    line_insert = f"{query_video_name},{target_video_name},{src_start_str},{src_end_str},{tgt_start_str},{tgt_end_str}\n"
                    annotation_lines.append(line_insert)
                    done_targets.add(target_video_name)

                    final_clip.close()
                    new_query_clip.close()

                    # 删除临时文件（异常处理后重试删除）
                    if os.path.exists(temp_query_path):
                        safe_remove(temp_query_path)

        # 保存当前分组的标注文件（将新旧内容合并写入）
        with open(annotation_path, "w", encoding="utf-8") as f:
            f.writelines(annotation_lines)
        print(f"Group {folder_num} done. Annotations saved to {annotation_path}")


if __name__ == "__main__":
    # 配置各个目录
    query_base_dir = r"D:\python workplace\NDVR\vcdb-diy\vcdb-diy-query\videos"
    target_base_dir = r"D:\python workplace\NDVR\vcdb-diy\vcdb-diy-origin"
    output_base_dir = r"D:\python workplace\NDVR\vcdb-diy\vcdb-diy-gen"
    annotation_dir = r"D:\python workplace\NDVR\vcdb-diy\vcdb-diy-query\annotations"

    process_videos(query_base_dir, target_base_dir, output_base_dir, annotation_dir, num_folders=10)
    print("所有分组处理完成！")

