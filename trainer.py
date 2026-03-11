import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import CustomPairDataset, FeatureProjector, CosineSimilarityLoss
from dataset_constructor import load_dataset
from util import *


# 验证模型
def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for featureA, featureB, label in val_loader:
            featureA, featureB, label = featureA.to(device), featureB.to(device), label.to(device)
            output1 = model(featureA)
            output2 = model(featureB)
            loss = criterion(output1, output2, label)
            running_loss += loss.item()
    return running_loss / len(val_loader)


# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    criterion.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for featureA, featureB, label in train_loader:
            featureA, featureB, label = featureA.to(device), featureB.to(device), label.to(device)

            optimizer.zero_grad()
            output1 = model(featureA)
            output2 = model(featureB)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = validate_model(model, val_loader, criterion, device)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}')


# 主训练函数
def main(dataset_path, save_path, epochs=10, batch=32, learning_rate=0.001, positive_threshold=0.9, negative_threshold=0.5):
    # 1. 加载数据集
    all_pairs = load_dataset(dataset_path)

    # 2. 划分数据集
    split_ratio = 0.8
    split_idx = int(len(all_pairs) * split_ratio)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    # 3. 创建数据加载器
    train_dataset = CustomPairDataset(train_pairs)
    val_dataset = CustomPairDataset(val_pairs)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    # 4. 初始化模型、损失函数、优化器
    input_dim = 512  # 根据预处理后的向量维度修改
    model = FeatureProjector(input_dim=input_dim, output_dim=1000)
    criterion = CosineSimilarityLoss(positive_threshold=positive_threshold, negative_threshold=negative_threshold)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 5. 设备检查
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 6. 训练模型
    num_epochs = epochs
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)

    save_name = save_path + 'trained_model_' + str(epochs) + '_' + str(positive_threshold) + '_' + str(negative_threshold) + '.pth'
    # 7. 保存模型
    torch.save(model.state_dict(), save_name)
    print("Model saved successfully.")


def feature_projection(feature_file, weights_path, save_path):
    features_data = load_features(feature_file)
    input_dim = 512  # 根据预处理后的向量维度修改
    feature_projector = FeatureProjector(input_dim=input_dim)
    feature_projector.load_state_dict(torch.load(weights_path))  # 加载之前训练好的权重
    feature_projector.eval()  # 设置为评估模式，不会进行参数更新
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_projector.to(device)
    for video_id, data in features_data.items():
        timestamps = data['shot_timestamps']
        features = data['shot_features']
        for i in range(len(features)):
            features[i] = torch.from_numpy(features[i])
            features[i] = features[i].to(device)  # 将特征移动到GPU
            features[i] = feature_projector(features[i])
        update_features(video_id, timestamps, features, save_path)
    print("All videos have been successfully processed.")


if __name__ == '__main__':
    # weights_path = 'models/epoch=10/trained_model_10_0.90_0.5.pth'
    # save_path = 'projected_features/k=1.5_10_0.9_0.5'
    # preprocessed_features = 'preprocessed_features/shot_features_core_v2_k=1.5'
    # feature_projection(preprocessed_features, weights_path, save_path)
    dataset_path = 'datasets/paired_dataset_k=1.0.pkl'
    save_path = 'models/epoch=10/'
    main(dataset_path, save_path, positive_threshold=0.85)
