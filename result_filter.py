import os

from results_analyse import parse_predictions


# 判断一个时间段是否完全包含另一个时间段
def is_contained(start1, end1, start2, end2):
    return start1 <= start2 and end1 >= end2


# 过滤结果，保留时间段较大的结果
def filter_predictions(predictions):
    filtered = []

    # 遍历每个预测结果
    for pred in predictions:
        videoA_pred, startA_pred, endA_pred, videoB_pred, startB_pred, endB_pred, confidence = pred

        should_add = True  # 标记是否应该加入过滤后的列表

        # 对过滤列表中的结果进行检查，防止包含关系
        for i, existing in enumerate(filtered):
            videoA_ex, startA_ex, endA_ex, videoB_ex, startB_ex, endB_ex, conf_ex = existing

            # 检查如果是同一个视频对
            if videoA_pred == videoA_ex and videoB_pred == videoB_ex:
                # 如果当前的结果包含在已有结果中，跳过该条
                if (is_contained(startA_ex, endA_ex, startA_pred, endA_pred) and
                        is_contained(startB_ex, endB_ex, startB_pred, endB_pred)):
                    should_add = False
                    break

                # 如果已有结果包含在当前的结果中，替换已有的结果
                elif (is_contained(startA_pred, endA_pred, startA_ex, endA_ex) and
                      is_contained(startB_pred, endB_pred, startB_ex, endB_ex)):
                    filtered[i] = pred
                    should_add = False
                    break

        # 如果该结果不包含在已有结果中，加入过滤后的列表
        if should_add:
            filtered.append(pred)

    return filtered


def filter(predictions_folder, save_folder):
    for pred_file in os.listdir(predictions_folder):
        if pred_file.endswith('.txt'):
            pred_path = os.path.join(predictions_folder, pred_file)
            predictions = parse_predictions(pred_path)
            filtered_predictions = filter_predictions(predictions)
            save_path = save_folder + pred_file
            with open(save_path, 'a') as file:
                for item in filtered_predictions:
                    videoA_pred, startA_pred, endA_pred, videoB_pred, startB_pred, endB_pred, confidence = item
                    formatted_result = f"{videoA_pred}\t{startA_pred}--{endA_pred}\t{videoB_pred}\t{startB_pred}--{endB_pred}\t{confidence}"
                    file.write(formatted_result + '\n')


if __name__ == '__main__':
    predictions_folder = 'prediction_results/multi-model(concate)'
    save_folder = 'prediction_results/multi-model(concate)-filter/'
    filter(predictions_folder, save_folder)
