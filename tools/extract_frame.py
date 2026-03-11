import cv2
import os

def extract_frames(video_path, output_folder):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频的帧率（FPS）
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    saved_frame_count = 0

    while True:
        # 读取视频的一帧
        ret, frame = cap.read()

        # 如果读取失败，说明视频结束，退出循环
        if not ret:
            break

        # 按每秒一帧的频率保存帧
        if frame_count % fps == 0:
            # 生成保存的文件名，格式为 frame_<编号>.jpg
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
            # 将帧保存为JPEG格式的图像
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    # 释放视频捕获对象
    cap.release()
    print(f"成功导出 {saved_frame_count} 帧到 {output_folder}")

# 示例调用
video_path = '3504e360accbaccb1580befbb441f1019664c2bb.mp4'  # 替换为你的视频文件路径
output_folder = 'frames'  # 替换为你想要保存帧的文件夹路径
extract_frames(video_path, output_folder)