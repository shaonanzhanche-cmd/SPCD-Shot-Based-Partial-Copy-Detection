import torch
import cv2
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset



# 帧特征提取模型
class FrameFeatureExtractor(nn.Module):
    def __init__(self):
        super(FrameFeatureExtractor, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.extractor = nn.Sequential(*list(self.model.children())[:-1])  # 移除原有的全连接层
        self.flatten = nn.Flatten()  # 添加 Flatten 层

    def forward(self, x):
        x = self.extractor(x)
        x = self.flatten(x)
        return x

# 镜头特征提取模型
class ShotFeatureExtractor(nn.Module):
    def __init__(self):
        super(ShotFeatureExtractor, self).__init__()
        self.model = models.video.mc3_18(pretrained=True)
        self.extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.extractor(x)
        x = self.flatten(x)
        return x


# 特征投影模型
class FeatureProjector(nn.Module):
    def __init__(self, input_dim=512, output_dim=1000):
        super(FeatureProjector, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 1024),  # 扩展维度到1024
            nn.BatchNorm1d(1024),       # 批量归一化
            nn.ReLU(inplace=True),      # 非线性激活函数
            nn.Linear(1024, output_dim)  # 投影到1000维
        )

    def forward(self, x):
        x = x.squeeze(1)
        return self.projector(x)


# 自定义数据集类
class CustomPairDataset(Dataset):
    def __init__(self, data_pairs):
        self.data_pairs = data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        featureA, featureB, label = self.data_pairs[idx]
        return torch.tensor(featureA, dtype=torch.float32), torch.tensor(featureB, dtype=torch.float32), torch.tensor(
            label, dtype=torch.float32)


# 基于余弦相似度的损失函数
class CosineSimilarityLoss(nn.Module):
    def __init__(self, positive_threshold=0.8, negative_threshold=0.2):
        super(CosineSimilarityLoss, self).__init__()
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, output1, output2, label):
        cos_sim = self.cosine_similarity(output1, output2)
        positive_loss = label * torch.pow(torch.clamp(self.positive_threshold - cos_sim, min=0.0), 2)
        negative_loss = (1 - label) * torch.pow(torch.clamp(cos_sim - self.negative_threshold, min=0.0), 2)
        loss = positive_loss + negative_loss
        return loss.mean()


# 端到端模型构建
class EndToEndModel(nn.Module):
    def __init__(self, feature_extractor, feature_projector):
        super(EndToEndModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.projector = feature_projector

    def forward(self, x):
        # 通过特征提取器提取特征
        features = self.feature_extractor(x)
        # 投影网络处理提取到的特征
        output = self.projector(features)
        return output


