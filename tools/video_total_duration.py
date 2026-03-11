import os
from moviepy.editor import VideoFileClip


def get_total_video_duration(folder_path):
    total_duration = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
                file_path = os.path.join(root, file)
                try:
                    clip = VideoFileClip(file_path)
                    total_duration += clip.duration
                    clip.close()
                except Exception as e:
                    print(f"读取文件 {file_path} 时出错: {e}")
    return total_duration


if __name__ == "__main__":
    folder = r'D:\python workplace\NDVR\vcdb-diy\vcdb-diy-query\videos'
    total_seconds = get_total_video_duration(folder)
    total_minutes = total_seconds / 60
    print(f"该文件夹下所有视频的总时长为 {total_minutes} 分钟。")

    # core 该文件夹下所有视频的总时长为 1615.539999999998 分钟。
    # core query 该文件夹下所有视频的总时长为    84.55950000000001    分钟。
    # diy query 该文件夹下所有视频的总时长为    # 13.125833333333334    # 分钟。