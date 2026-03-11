import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from util import *
from sklearn.manifold import TSNE


original_features = 'preprocessed_features/shot_features_core_v2_k=1.5'
projected_features = 'projected_features/k=1.5_50_0.9_0.5——k=1.0'
original_features_data = load_features(original_features)
projected_features_data = load_features(projected_features)
video1_id = '3504e360accbaccb1580befbb441f1019664c2bb.mp4'
video2_id = '37b31d607d31a47d347b15dae2b8aa63e57861eb.flv'
video1_original = original_features_data[video1_id]['shot_features']
video2_original = original_features_data[video2_id]['shot_features']
video1_projected = projected_features_data[video1_id]['shot_features']
video2_projected = projected_features_data[video2_id]['shot_features']
print(len(video1_original))
print(len(video2_original))

# 定义一个函数来调整特征列表的格式
def adjust_format(feature_list, expected_dim):
    # 检查列表中的每个元素是否为指定维度的特征向量
    for i, feature in enumerate(feature_list):
        if not isinstance(feature, np.ndarray):
            feature = np.array(feature)
        if feature.ndim == 1:
            # 如果是一维数组，将其转换为二维数组
            feature = feature.reshape(1, -1)
        elif feature.ndim != 2 or feature.shape[1] != expected_dim:
            raise ValueError(f"Feature at index {i} has an incorrect shape. Expected shape: (1, {expected_dim})")
        feature_list[i] = feature
    # 堆叠数组
    return np.vstack(feature_list)

# 调整原始特征的格式（1×512 维）
video1_original = adjust_format(video1_original, 512)
video2_original = adjust_format(video2_original, 512)

# 调整投影特征的格式（1×1000 维）
video1_projected = adjust_format(video1_projected, 1000)
video2_projected = adjust_format(video2_projected, 1000)
# 假设video1_original、video2_original是两个视频的原始镜头特征向量列表
# video1_projected、video2_projected是投影后的镜头特征向量列表
distance_matrix_original = cosine_distances(np.vstack([video1_original, video2_original]))
distance_matrix_projected = cosine_distances(np.vstack([video1_projected, video2_projected]))

# 确定统一的 vmin 和 vmax
combined_distances = np.hstack([distance_matrix_original.flatten(), distance_matrix_projected.flatten()])
vmin = np.min(combined_distances)
vmax = np.max(combined_distances)

plt.figure(figsize=(12, 6))

# 获取视频 1 的镜头数量，用于确定交接位置
video1_shot_num_original = len(video1_original)
video1_shot_num_projected = len(video1_projected)

# 绘制原始特征距离矩阵热力图
ax1 = plt.subplot(1, 2, 1)
sns.heatmap(distance_matrix_original, cmap='YlGnBu', vmin=vmin, vmax=vmax)
# 调整标题位置，pad 表示标题与子图的间距
ax1.set_title('Original Feature', pad=20)
# 绘制垂直分割线
plt.axvline(x=video1_shot_num_original, color='r', linestyle='--', linewidth=2)
# 绘制水平分割线
plt.axhline(y=video1_shot_num_original, color='r', linestyle='--', linewidth=2)
# 添加 video1 和 video2 标注在标题下方
total_width = len(video1_original) + len(video2_original)
ax1.text(video1_shot_num_original / 2, -0.5, 'Video A', ha='center', va='center', fontsize=10)
ax1.text(video1_shot_num_original + len(video2_original) / 2, -0.5, 'Video B', ha='center', va='center', fontsize=10)

# 绘制投影特征距离矩阵热力图
ax2 = plt.subplot(1, 2, 2)
sns.heatmap(distance_matrix_projected, cmap='YlGnBu', vmin=vmin, vmax=vmax)
# 调整标题位置，pad 表示标题与子图的间距
ax2.set_title('Projected Feature', pad=20)
# 绘制垂直分割线
plt.axvline(x=video1_shot_num_projected, color='r', linestyle='--', linewidth=2)
# 绘制水平分割线
plt.axhline(y=video1_shot_num_projected, color='r', linestyle='--', linewidth=2)
# 添加 video1 和 video2 标注在标题下方
total_width_projected = len(video1_projected) + len(video2_projected)
ax2.text(video1_shot_num_projected / 2, -0.5, 'Video A', ha='center', va='center', fontsize=10)
ax2.text(video1_shot_num_projected + len(video2_projected) / 2, -0.5, 'Video B', ha='center', va='center', fontsize=10)

plt.show()
