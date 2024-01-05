import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from depth_model import DepthEstimationNet

from test_depth_dataset import NYUDepthTestDataset

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    root_dir = 'nyu2_test'
    transform = transforms.Compose([transforms.ToTensor()])  # Define your transformations here

    model = DepthEstimationNet()

    model_path = 'model_state_dict_9.pth'


    model.load_state_dict(torch.load(model_path))

    model.eval()
    print("Loaded model...")

    dataset = NYUDepthTestDataset(root_dir, transform=transform)
    print("Dataset loaded.")

    for item in dataset:
        rgb = item[0]
        rgb_image = rgb.squeeze(0).detach().numpy()
        rgb_valid = rgb.unsqueeze(0)
        rgb_image = np.transpose(rgb_image, (1, 2, 0))
        predicted_depth = model(rgb_valid)
        predicted_depth = predicted_depth.squeeze(0)
        depth_image = predicted_depth.squeeze(0).detach().numpy()

        print(depth_image.shape)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Display the RGB image
        axes[0].imshow(rgb_image)
        axes[0].set_title('RGB Image')
        axes[0].axis('off')  # Turn off axis numbers

        # Display the depth image
        axes[1].imshow(depth_image, cmap='gray')
        axes[1].set_title('Depth Image')
        axes[1].axis('off')  # Turn off axis numbers
        plt.show()
