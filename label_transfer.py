from datetime import datetime, timedelta
from util import *


# Helper function to convert time string to seconds
def time_str_to_seconds(time_str):
    time_obj = datetime.strptime(time_str, "%H:%M:%S")
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second


# Helper function to convert seconds to time string
def seconds_to_time_str(seconds):
    return str(timedelta(seconds=seconds))


def jaccard_index_time(interval1, interval2):
    # 计算交集的开始和结束时间
    start_intersection = max(interval1[0], interval2[0])
    end_intersection = min(interval1[1], interval2[1])

    # 计算并集的开始和结束时间
    start_union = min(interval1[0], interval2[0])
    end_union = max(interval1[1], interval2[1])

    # 计算交集和并集的持续时间
    duration_intersection = max(0, end_intersection - start_intersection)
    duration_union = end_union - start_union

    # 计算交并比
    if duration_union == 0:
        return 0  # 避免除以零的情况
    jaccard_index = duration_intersection / duration_union

    return jaccard_index


def normalize_shot(start, end, shots_timestamp):
    duration = end - start
    normalized_timestamp = [0]
    offset = 0
    for index in range(len(shots_timestamp) - 1):
        if shots_timestamp[index] <= start < shots_timestamp[index + 1]:
            offset = index
            break
    for i in range(offset, len(shots_timestamp) - 1):
        if start < shots_timestamp[i] < end:
            normalized_timestamp.append((shots_timestamp[i] - start) / duration * 100)
    normalized_timestamp.append(100)
    return offset, normalized_timestamp


# Function to process tags from file and save to new file
def process_tags(input_file, output_file, feature_path):
    labels = []
    feature_data = load_features(feature_path)
    with open(input_file, 'r') as file:
        for line in file:
            new_tags = []
            video_a, video_b, start_a_str, end_a_str, start_b_str, end_b_str = line.strip().split(',')
            start_a = time_str_to_seconds(start_a_str)
            end_a = time_str_to_seconds(end_a_str)
            start_b = time_str_to_seconds(start_b_str)
            end_b = time_str_to_seconds(end_b_str)
            video_a_shots = feature_data[video_a]['shot_timestamps']
            video_b_shots = feature_data[video_b]['shot_timestamps']
            if video_a_shots[0] != 0:
                video_a_shots.insert(0, 0)
            if video_b_shots[0] != 0:
                video_b_shots.insert(0, 0)

            offset_a, normalized_a_shot = normalize_shot(start_a, end_a, video_a_shots)
            offset_b, normalized_b_shot = normalize_shot(start_b, end_b, video_b_shots)
            for i in range(len(normalized_a_shot) - 1):
                shot_a = (normalized_a_shot[i], normalized_a_shot[i + 1])
                for j in range(len(normalized_b_shot) - 1):
                    shot_b = (normalized_b_shot[j], normalized_b_shot[j + 1])
                    if jaccard_index_time(shot_a, shot_b) >= 0.5:
                        new_tag = (offset_a + i, offset_b + j)
                        new_tags.append(new_tag)
            label_item = (video_a, video_b, new_tags)
            labels.append(label_item)

    with open(output_file, 'w') as file:
        for label_item in labels:
            file.write(label_item[0] + ',' + label_item[1] + '\t')
            for item in label_item[2]:
                file.write(f"{item}")
            file.write('\n')


def process_all(root_path, output_path, feature_path):
    annotation_files = annotation_loader(root_path)
    for annotation_file in annotation_files:
        output_file = output_path + annotation_file.split('\\')[-1]
        process_tags(annotation_file, output_file, feature_path)


if __name__ == "__main__":
    # Example usage
    feature_path = "preprocessed_features/shot_features_core_v2_k=1.0"
    root_path = "D:\\python workplace\\NDVR\\vcdb-core\\annotation"
    output_path = './annotations_k=1.0/'
    process_all(root_path, output_path, feature_path)
