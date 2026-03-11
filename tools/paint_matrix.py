import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.colors as mcolors
from util import *


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

projected_features = 'projected_features/k=1.5_50_0.85'
projected_features_data = load_features(projected_features)
video1_id = '3504e360accbaccb1580befbb441f1019664c2bb.mp4'
video2_id = '37b31d607d31a47d347b15dae2b8aa63e57861eb.flv'
video1_projected = projected_features_data[video1_id]['shot_features']
video2_projected = projected_features_data[video2_id]['shot_features']
print(len(video1_projected))
print(len(video2_projected))

# 调整投影特征的格式（1×1000 维）
video1_projected = adjust_format(video1_projected, 1000)
video2_projected = adjust_format(video2_projected, 1000)

# 计算余弦距离矩阵
distance_matrix_projected = cosine_distances(np.vstack([video1_projected, video2_projected]))

# 计算余弦相似度矩阵
similarity_matrix_projected = 1 - distance_matrix_projected

# 确定 vmin 和 vmax
vmin = np.min(similarity_matrix_projected)
vmax = np.max(similarity_matrix_projected)

# 设置图形窗口大小为正方形
fig = plt.figure(figsize=(8, 8))

# 获取视频 1 的镜头数量，用于确定交接位置
video1_shot_num_projected = len(video1_projected)

# 自定义颜色映射，将相似度大于 0.9 的区域设置为红色
cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', [(0, 'lightblue'), (0.85, 'lightblue'), (0.85, 'red'), (1, 'red')])

# 绘制投影特征相似度矩阵热力图，并获取颜色条对象
im = sns.heatmap(similarity_matrix_projected, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={"fraction": 0.046, "pad": 0.04})
cbar = im.collections[0].colorbar

# 调整标题位置，pad 表示标题与子图的间距
ax = im.axes
ax.set_title('T_p=0.85', pad=20)
# 绘制垂直分割线
plt.axvline(x=video1_shot_num_projected, color='r', linestyle='--', linewidth=2)
# 绘制水平分割线
plt.axhline(y=video1_shot_num_projected, color='r', linestyle='--', linewidth=2)
# 添加 video1 和 video2 标注在标题下方
total_width_projected = len(video1_projected) + len(video2_projected)
ax.text(video1_shot_num_projected / 2, -0.5, 'Video A', ha='center', va='center', fontsize=10)
ax.text(video1_shot_num_projected + len(video2_projected) / 2, -0.5, 'Video B', ha='center', va='center', fontsize=10)

# 使坐标轴等比例，确保热力图为正方形
ax.set_aspect('equal', adjustable='box')

# 调整颜色条的高度，使其与主图一致
pos = cbar.ax.get_position()
pos.y0 = ax.get_position().y0
pos.y1 = ax.get_position().y1
cbar.ax.set_position(pos)

plt.tight_layout()

# 保存图像
save_path = 'projected_feature_similarity_heatmap_0.85.png'
plt.savefig(save_path, dpi=300)

plt.show()