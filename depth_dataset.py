import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class NYUDepthDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the RGB and depth images.
            depth_dir (string): Directory with all the Depth images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.folders = os.listdir(root_dir)

        self.filenames = []
        self.get_all_filenames()

        # get paths to every example

    def get_all_filenames(self):

        for folder in self.folders:
            dir_name = os.path.join(self.root_dir, folder)
            files = os.listdir(dir_name)
            current_val = None
            for i in range(0, len(files), 2):
                file1 = files[i]
                file2 = files[i+1]
                # jpg is rgb, png is depth
                tup = (os.path.join(dir_name, file1), os.path.join(dir_name, file2))
                if 'jpg' in file2:
                    tup = (tup[1], tup[0])

                self.filenames.append(tup)


    def __len__(self):
        # to get length, go through each folder in dataset
        total_length = 0
        for folder in self.folders:
            dir_name = os.path.join(self.root_dir, folder)
            files = os.listdir(dir_name)
            num_examples = int(len(files) / 2)

            total_length += num_examples

        # return total_length
        return 5000

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rgb_img_name = self.filenames[idx][0]
        depth_img_name = self.filenames[idx][1]

        rgb_image = Image.open(rgb_img_name).convert('RGB')
        depth_image = Image.open(depth_img_name).convert('L')  # Assuming depth images are grayscale

        if self.transform:
            rgb_image = self.transform(rgb_image)
            depth_image = self.transform(depth_image)

        return rgb_image, depth_image

# Example usage
if __name__ == '__main__':
    root_dir = 'nyu2_train'
    rgb_dir = 'nyu2_train'
    depth_dir = 'path_to_depth_images'
    transform = transforms.Compose([transforms.ToTensor()])  # Define your transformations here

    dataset = NYUDepthDataset(root_dir, transform=transform)

    rgb, depth = dataset[0]

    print(rgb.shape)
    print(depth.shape)

