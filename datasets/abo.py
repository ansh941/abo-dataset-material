import os
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset

import torchvision.transforms as transforms

import cv2
from PIL import Image

class SVABOMaterialDataset(Dataset):
    def __init__(self, root_dir, image_transform=None, label_transform=None, train=True):
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.label_transform = label_transform

        # Load the data
        # Load file that contains the indices of the render views to be used for each env map
        if train:
            with open(os.path.join(self.root_dir, 'train_sample_idx.json'), 'r') as f:
                train_sample_idx = json.load(f)
        else:
            with open(os.path.join(self.root_dir, 'test_sample_idx.json'), 'r') as f:
                train_sample_idx = json.load(f)
        
        self.render_view_images = []
        self.base_color_images = []
        self.normal_images = []
        self.metallic_roughness_images = []
        self.mask_images = []
        for model_id, env_map_ids in train_sample_idx.items():
            for env_map_id, render_view_ids in env_map_ids.items():
                for render_view_id in render_view_ids:
                    self.render_view_images.append(os.path.join(self.root_dir, model_id, 'render', env_map_id, f'render_{render_view_id}.jpg'))
                    self.base_color_images.append(os.path.join(self.root_dir, model_id, 'base_color', f'base_color_{render_view_id}.jpg'))
                    self.normal_images.append(os.path.join(self.root_dir, model_id, 'normal', f'normal_{render_view_id}.png'))
                    self.metallic_roughness_images.append(os.path.join(self.root_dir, model_id, 'metallic_roughness', f'metallic_roughness_{render_view_id}.jpg'))
                    self.mask_images.append(os.path.join(self.root_dir, model_id, 'segmentation', f'segmentation_{render_view_id}.jpg'))
    def __len__(self):
        return len(self.render_view_images)

    def __getitem__(self, idx):
        render_view_img = Image.open(self.render_view_images[idx])
        base_color_img = Image.open(self.base_color_images[idx])
        normal_img = Image.open(self.normal_images[idx])
        
        # R: Metallic, G: Roughness, B: Unused
        metallic_roughness_img = Image.open(self.metallic_roughness_images[idx])

        mask_img = Image.open(self.mask_images[idx])
        
        if self.image_transform:
            render_view_img = self.image_transform(render_view_img)
        
        if self.label_transform:
            base_color_img = self.label_transform(base_color_img)
            normal_img = self.label_transform(normal_img)
            metallic_roughness_img = self.label_transform(metallic_roughness_img)
            mask_img = self.label_transform(mask_img)
            
        return render_view_img, base_color_img, normal_img, metallic_roughness_img, mask_img

if __name__ == '__main__':
    root_dir = '../../datasets/abo-benchmark-material'
    
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((512, 512)),
    ])
    
    label_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512))
    ])
    dataset = SVABOMaterialDataset(root_dir, image_transform=image_transform, label_transform=label_transforms, train=True)
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for i, data in enumerate(dataloader):
        render_view_img, base_color_img, normal_img, metallic_roughness_img = data
        print(render_view_img.shape, base_color_img.shape, normal_img.shape, metallic_roughness_img.shape)
        break
    
    