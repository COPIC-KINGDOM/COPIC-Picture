import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained VGG model and move it to GPU
vgg_model = models.vgg16(pretrained=True)
vgg_model = vgg_model.to(device)

# Define a new model with VGG's convolutional layers
vgg_features = vgg_model.features
vgg_features.eval()

# Define transformation and load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cifar10_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
data_loader = DataLoader(cifar10_dataset, batch_size=1, shuffle=True)

# Iterate over each label
for label in range(10):
    # Define output and heatmap folders for the current label
    output_folder = f'/media/tust/COPICAI/cifar_data/cifar_{label}_vgg_outputs'
    heatmap_folder = f'/media/tust/COPICAI/cifar_data_img/cifar_{label}_hot_img'
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(heatmap_folder, exist_ok=True)

    # Iterate over each image in the dataset
    for idx, batch in enumerate(data_loader):
        labels = batch[1]
        if labels.item() == label:
            input_image = batch[0].to(device)
            all_outputs = {}

            # Iterate over each layer and save outputs
            for i, layer in enumerate(vgg_features):
                input_image = layer(input_image)
                layer_output = input_image.squeeze().detach().cpu().numpy()

                for j in range(layer_output.shape[0]):
                    key = f'vgg_layer_{i + 1}_channel_{j + 1}_output'
                    all_outputs[key] = layer_output[j:j + 1]

                    # Optionally, save heatmap images
                    # channel_output = layer_output[j:j + 1].squeeze()
                    # plt.imshow(channel_output, cmap='viridis', aspect='auto')
                    # plt.title(f'Layer {i + 1} Channel {j + 1} Output')
                    # plt.colorbar()
                    # heatmap_filename = os.path.join(heatmap_folder,
                    #                                 f'vgg_layer_{i + 1}_channel_{j + 1}_output_heatmap_{idx}.png')
                    # plt.savefig(heatmap_filename)
                    # plt.close()

            npz_filename = os.path.join(output_folder, f'image_{idx}_npz.npz')
            np.savez(npz_filename, **all_outputs)
            print(f"Label {label}, Image {idx}: Outputs saved to {npz_filename}")

            del input_image, all_outputs
            torch.cuda.empty_cache()
