import os
from PIL import Image

import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

import config

class ZeroPad_Resize:
    """
    Zero padds the image to a square of shape (dataset_image_size, dataset_image_size)
    Then it's reshaped to shape (input_image_size, input_image_size)
    """
    def __init__(self):
        self.dataset_image_size = config.dataset_image_size
        self.input_image_size = config.input_image_size

    def __call__(self, img):
        pad_height = max(0, self.dataset_image_size[0] - img.shape[1])
        pad_width  = max(0, self.dataset_image_size[1] - img.shape[2])

        top_pad = pad_height // 2
        bottom_pad = pad_height - top_pad
        left_pad = pad_width // 2
        right_pad = pad_width - left_pad
        
        img = nn.functional.pad(img, (left_pad, right_pad, top_pad, bottom_pad), mode='constant', value=0)
        img = transforms.functional.resize(img, (self.input_image_size))

        return img


class DefocusDataset(Dataset):
    def __init__(self, root_dir_images, root_dir_labels):
        self.root_dir_images = root_dir_images
        self.root_dir_labels = root_dir_labels
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            ZeroPad_Resize()
        ])

        self.image_files = sorted(os.listdir(root_dir_images))
        self.label_files = sorted(os.listdir(root_dir_labels))

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir_images, self.image_files[index])
        label_path = os.path.join(self.root_dir_labels, self.label_files[index])

        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")

        image = self.transform(image)
        label = self.transform(label)

        return {'image': image, 'label': label}

    def __len__(self):
        return len(self.image_files)
