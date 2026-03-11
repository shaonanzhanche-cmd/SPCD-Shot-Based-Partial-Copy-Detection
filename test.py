# vcdb core:
# Tp=0.9  Tn=0.5
# T=0.9
# Sp=0.7393  Sr=0.9364
# F1=0.8263
#
#
# Total Precision: 0.7393
# Total Recall: 0.9364
# Total Precision: 33340.0
# Total Recall: 45094
# Total Precision: 663.0
# Total Recall: 708
#
# distraction1 = 647  0.7289
# distraction2 = 728  0.7175
# distraction3 = 614  0.7081
#
# without_fp:
# Total Precision: 0.5231
# Total Recall: 0.7811
# F1 = 0.6266
# tp=0.85:
# Total Precision: 0.7200
# Total Recall: 0.9492
# tp=0.90:
#
# tp=0.95:
# Total Precision: 0.8065
# Total Recall: 0.8757
# k=1.0:
# Total Precision: 0.7399
# Total Recall: 0.9718
# Given precision and recall values
precision = 0.7345
recall = 0.9336

# F1 score calculation
f1_score = 2 * (precision * recall) / (precision + recall)
print(f1_score)
# import torch
# print(torch.cuda.is_available())
# [1:00:34<00:00,  6.88s/it]
# from util import *
#
# features_path = 'preprocessed_features/shot_segmentation_para/k=1.5_W=15_I=10'
# features_data = load_features(features_path)
# total_time = 0
# total_shot = 0
# i = 1
# for video_id, data in features_data.items():
#     print(f"Video number: {i}")
#     print(f"Video ID: {video_id}")
#     print(f"Shot Timestamps: {data['shot_timestamps']}")
#     print(f"Shot Features Length: {len(data['shot_features'])}")
#     shot_num = len(data['shot_features'])
#     video_len = data['shot_timestamps'][shot_num-1]
#     total_shot += shot_num
#     total_time += video_len
#     i += 1
# print(f"总镜头数：{total_shot}")
# print(f"总时长：{total_time}")
# k=1.0 I=10 W=10
# 总镜头数：31536
# 总时长：95111
# k=1.5 I=10 W=10
# 总镜头数：21914
# 总时长：95128
# k=2.0 I=10 W=10
# 总镜头数：12634
# 总时长：95137
# k=1.5 I=15 W=10
# 总镜头数：15723
# 总时长：95138
# k=1.5 I=5 W=10
# 总镜头数：36233
# 总时长：95080
# k=1.5 I=10 W=15
# 总镜头数：20180
# 总时长：95138
# k=1.5 I=10 W=5
# 总镜头数：26179
# 总时长：95113


# retrieving videos: 100%|██████████| 28/28 [28:45<00:00, 61.63s/it]
# retrieving videos: 100%|██████████| 28/28 [30:20<00:00, 65.00s/it] k=2.0/projected
# retrieving videos: 100%|██████████| 28/28 [1:18:25<00:00, 168.05s/it] k=1.5/projected
# retrieving videos: 100%|██████████| 10/10 [03:06<00:00, 18.63s/it] k=1.0/diy
# retrieving videos: 100%|██████████| 10/10 [01:30<00:00,  9.06s/it] k=2.0/diy
# retrieving videos: 100%|██████████| 10/10 [02:11<00:00, 13.16s/it] tp=0.95/diy
# retrieving videos: 100%|██████████| 10/10 [02:10<00:00, 13.04s/it]