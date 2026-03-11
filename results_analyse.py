import os
from datetime import timedelta


# 将时间戳 "hh:mm:ss" 转换为 timedelta 对象
def parse_timestamp(timestamp):
    h, m, s = map(int, timestamp.split(':'))
    return timedelta(hours=h, minutes=m, seconds=s)


# 检查两个时间段是否有交集
def has_intersection(start1, end1, start2, end2):
    return max(start1, start2) < min(end1, end2)


# 解析预测结果
def parse_predictions(pred_file):
    predictions = []
    with open(pred_file, 'r') as f:
        for line in f:
            if line.strip():
                videoA, timeA, videoB, timeB, confidence = line.strip().split('\t')
                startA, endA = timeA.split('--')
                startB, endB = timeB.split('--')
                predictions.append((videoA, parse_timestamp(startA), parse_timestamp(endA),
                                    videoB, parse_timestamp(startB), parse_timestamp(endB),
                                    float(confidence)))
    return predictions


# 解析标注数据
def parse_ground_truth(gt_file):
    ground_truth = []
    with open(gt_file, 'r') as f:
        for line in f:
            if line.strip():
                videoA, videoB, startA, endA, startB, endB = line.strip().split(',')
                ground_truth.append((videoA, parse_timestamp(startA), parse_timestamp(endA),
                                     videoB, parse_timestamp(startB), parse_timestamp(endB)))
    return ground_truth


# 计算 Precision 和 Recall
def calculate_precision_recall(predictions, ground_truth):
    true_positives_precision = 0
    true_positives_recall = 0

    # 计算 true positives for Precision
    for pred in predictions:
        videoA_pred, startA_pred, endA_pred, videoB_pred, startB_pred, endB_pred, _ = pred
        for gt in ground_truth:
            videoA_gt, startA_gt, endA_gt, videoB_gt, startB_gt, endB_gt = gt
            if ((videoA_pred == videoA_gt and videoB_pred == videoB_gt and
                 has_intersection(startA_pred, endA_pred, startA_gt, endA_gt) and
                 has_intersection(startB_pred, endB_pred, startB_gt, endB_gt)) or
                    (videoA_pred == videoB_gt and videoB_pred == videoA_gt and
                     has_intersection(startA_pred, endA_pred, startB_gt, endB_gt) and
                     has_intersection(startB_pred, endB_pred, startA_gt, endA_gt))):
                true_positives_precision += 1
                break  # 一条预测只能对应一个真实值

    # 计算 true positives for Recall
    for gt in ground_truth:
        videoA_gt, startA_gt, endA_gt, videoB_gt, startB_gt, endB_gt = gt
        for pred in predictions:
            videoA_pred, startA_pred, endA_pred, videoB_pred, startB_pred, endB_pred, _ = pred
            if ((videoA_pred == videoA_gt and videoB_pred == videoB_gt and
                 has_intersection(startA_pred, endA_pred, startA_gt, endA_gt) and
                 has_intersection(startB_pred, endB_pred, startB_gt, endB_gt)) or
                    (videoA_pred == videoB_gt and videoB_pred == videoA_gt and
                     has_intersection(startA_pred, endA_pred, startB_gt, endB_gt) and
                     has_intersection(startB_pred, endB_pred, startA_gt, endA_gt))):
                true_positives_recall += 1
                break  # 一条真实值只能被一条预测匹配

    # Precision = true positives for predictions / total predictions
    precision = true_positives_precision / len(predictions) if predictions else 0

    # Recall = true positives for ground truth / total ground truth
    recall = true_positives_recall / len(ground_truth) if ground_truth else 0

    return precision, recall


# 读取文件夹中的所有txt文件并计算 Precision 和 Recall
def process_folder(predictions_folder, ground_truth_folder):
    total_true_positives_precision = 0
    total_true_positives_recall = 0
    total_predictions = 0
    total_ground_truth = 0

    for pred_file in os.listdir(predictions_folder):
        if pred_file.endswith('.txt'):
            pred_path = os.path.join(predictions_folder, pred_file)
            gt_path = os.path.join(ground_truth_folder, pred_file)  # 假设文件名一致

            if os.path.exists(gt_path):
                predictions = parse_predictions(pred_path)
                ground_truth = parse_ground_truth(gt_path)

                precision, recall = calculate_precision_recall(predictions, ground_truth)

                # 按每个文件内的条目数加权
                total_true_positives_precision += precision * len(predictions)
                total_true_positives_recall += recall * len(ground_truth)
                total_predictions += len(predictions)
                total_ground_truth += len(ground_truth)

    # 计算整体的 Precision 和 Recall
    final_precision = total_true_positives_precision / total_predictions if total_predictions > 0 else 0
    final_recall = total_true_positives_recall / total_ground_truth if total_ground_truth > 0 else 0

    return final_precision, final_recall, total_true_positives_precision, total_predictions, total_true_positives_recall, total_ground_truth


def count_results(folder_path):
    count = 0
    for pred_file in os.listdir(folder_path):
        pred_path = os.path.join(folder_path, pred_file)
        with open(pred_path, 'r') as f:
            for line in f:
                if line.strip():
                    count += 1
    return count


# 主函数
def main():
    predictions_folder = 'prediction_results/distraction3-black_filter'  # 预测结果文件夹
    ground_truth_folder = r'D:\python workplace\NDVR\vcdb-core-test\annotation'  # 真实标注文件夹

    precision, recall, total_true_positives_precision, total_predictions, total_true_positives_recall, total_ground_truth = process_folder(predictions_folder, ground_truth_folder)
    # f1_score = 2 * (precision * recall) / (precision + recall)

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'True_Positive_prediction: {total_true_positives_precision}')
    print(f'Total Prediction: {total_predictions}')
    print(f'True_Positive_recall: {total_true_positives_recall}')
    print(f'Total Recall: {total_ground_truth}')
    # print(f'F1-score:{f1_score}')


if __name__ == '__main__':
    main()
    # folder = 'distraction3'
    # count = count_results(folder)
    # print(count)
