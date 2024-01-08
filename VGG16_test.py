# VGG训练cifar-10数据集，遍历cifar-10，每张图片按照每层按照通道数生成热图数据，每层数据保存成一个npz文件（GPU）
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的 VGG 模型，并将其移动到 GPU
vgg_model = models.vgg16(pretrained=True)
vgg_model = vgg_model.to(device)

# 定义一个新的模型，包含 VGG 的前半部分（卷积层）
vgg_features = vgg_model.features

# 将模型设置为评估模式
vgg_features.eval()

# 选择输出文件夹
output_folder = '/media/tust/COPICAI/cifar_vgg_outputs_channel'
heatmap_folder = '/media/tust/COPICAI/cifar_hot_img_channel'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(heatmap_folder, exist_ok=True)

# 加载 CIFAR-10 数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cifar10_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
data_loader = DataLoader(cifar10_dataset, batch_size=1, shuffle=True)

# 遍历每张图片并生成相应的 npz 文件
for idx, batch in enumerate(data_loader):
    input_image = batch[0].to(device)

    # 创建一个字典用于存储当前图片的所有通道的输出
    all_outputs = {}

    # 遍历每一层并保存每个通道的输出和热图
    for i, layer in enumerate(vgg_features):
        input_image = layer(input_image)

        layer_output = input_image.squeeze().detach().cpu().numpy()  # 将输出移动回 CPU
        for j in range(layer_output.shape[0]):  # 遍历通道
            # 生成唯一的键，用于存储每个通道的输出
            key = f'vgg_layer_{i + 1}_channel_{j + 1}_output'
            all_outputs[key] = layer_output[j:j + 1]

            # 绘制热图
            # channel_output = layer_output[j:j + 1].squeeze()  # 去除维度为 1 的维度
            # plt.imshow(channel_output, cmap='viridis', aspect='auto')
            # plt.title(f'Layer {i + 1} Channel {j + 1} Output')
            # plt.colorbar()
            # heatmap_filename = os.path.join(heatmap_folder,
            #                                 f'vgg_layer_{i + 1}_channel_{j + 1}_output_heatmap_{idx}.png')
            # plt.savefig(heatmap_filename)
            # plt.close()

            # print(f"第 {i + 1} 层 第 {j + 1} 通道：输出形状 - {channel_output.shape}，已保存到 {heatmap_filename}")

    # 将当前图片的所有通道的输出保存到一个名为 'image_{idx}_npz.npz' 的文件中
    npz_filename = os.path.join(output_folder, f'image_{idx}_npz.npz')
    np.savez(npz_filename, **all_outputs)
    print(f"第 {idx} 张图片的所有通道的输出已保存到 {npz_filename}")

    # 删除不再需要的对象，以释放内存
    del input_image, all_outputs, layer_output
    torch.cuda.empty_cache()  # 释放 GPU 存储