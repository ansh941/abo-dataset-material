import os
import json

import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset

import torchvision.transforms as transforms

import cv2
from PIL import Image

from rendering_test import render

def create_hdr_image_list(root_dir):
    hdr_image_list = []
    for model_id in tqdm.tqdm(os.listdir(root_dir)):
        if os.path.isdir(os.path.join(root_dir, model_id)):
            print(f'Processing {model_id}')
            with open(os.path.join(root_dir, model_id, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
            

            hdr_image_list.extend(metadata['envs'])
    hdr_image_list = list(set(hdr_image_list))
    
    return hdr_image_list

def download_hdr_images(hdr_image_list):
    url_root = 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/'
    for hdr_image in tqdm.tqdm(hdr_image_list):
        os.system(f'wget -P {os.path.join('.', 'env_maps')} {url_root}{hdr_image}')
class SVABOMaterialDataset(Dataset):
    def __init__(self, root_dir, image_transform=None, label_transform=None, train=True):
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.label_transform = label_transform

        # Load the data
        # Load file that contains the indices of the render views to be used for each env map
        if train:
            with open(os.path.join(self.root_dir, 'train_sample_idx.json'), 'r') as f:
                sample_idx = json.load(f)
        else:
            with open(os.path.join(self.root_dir, 'test_sample_idx.json'), 'r') as f:
                sample_idx = json.load(f)
        
        self.render_view_images = []
        self.base_color_images = []
        self.normal_images = []
        self.metallic_roughness_images = []
        self.mask_images = []
        for model_id, env_map_ids in sample_idx.items():
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
        render_view = cv2.imread(self.render_view_images[idx], 1) / 255
        base_color = cv2.imread(self.base_color_images[idx], 1) / 255
        normal = cv2.imread(self.normal_images[idx], 1)
        normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
        normal = (normal / 255.0) * 2.0 - 1.0  # Normalize to [-1, 1]
        
        # B: Metallic, G: Roughness, R: Unused
        # metallic_roughness = Image.open(self.metallic_roughness_images[idx])
        metallic_roughness = cv2.imread(self.metallic_roughness_images[idx], 1) / 255
        metallic = metallic_roughness[:, :, 0:1]
        roughness = metallic_roughness[:, :, 1:2]
        
        mask = cv2.imread(self.mask_images[idx], 0) / 255
        
        recon_view = render(base_color, metallic, roughness, normal, np.array([0, 0, 1]), np.array([0, 0, 1]))
        
        if self.image_transform:
            render_view = self.image_transform(render_view)
        
        if self.label_transform:
            base_color = self.label_transform(base_color)
            normal = self.label_transform(normal)
            metallic = self.label_transform(metallic)
            roughness = self.label_transform(roughness)
            mask = self.label_transform(mask)
            recon_view = self.label_transform(recon_view)
            
        return render_view, base_color, normal, metallic, roughness, mask, recon_view
    
if __name__ == '__main__':
    root_dir = '../../datasets/abo-benchmark-material'
    hdr_image_list = create_hdr_image_list(root_dir)
    download_hdr_images(hdr_image_list)
    '''
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
    '''