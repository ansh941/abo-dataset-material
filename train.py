import os

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorboard

from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.abo import SVABOMaterialDataset
from models.svnet import SVNet

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def visualize(render_view_img, base_color_img, normal_img, metallic_roughness_img, mask_img, base_color_pred, normal_pred, metallic_pred, roughness_pred, mean, std):
    render_view_img = (render_view_img * std + mean).detach().cpu().numpy().transpose(0,2,3,1)[0]*255
    base_color_img = base_color_img.detach().cpu().numpy().transpose(0,2,3,1)[0]*255
    normal_img = normal_img.detach().cpu().numpy().transpose(0,2,3,1)[0]*255
    metallic_img = cv2.cvtColor(metallic_roughness_img[:,2:3].detach().cpu().numpy().transpose(0,2,3,1)[0]*255, cv2.COLOR_GRAY2BGR)
    roughness_img = cv2.cvtColor(metallic_roughness_img[:,1:2].detach().cpu().numpy().transpose(0,2,3,1)[0]*255, cv2.COLOR_GRAY2BGR)
    
    base_color_pred = base_color_pred.detach().cpu().numpy().transpose(0,2,3,1)[0]*255
    normal_pred = normal_pred.detach().cpu().numpy().transpose(0,2,3,1)[0]*255
    metallic_pred = metallic_pred.detach().cpu().numpy().transpose(0,2,3,1)[0]*255
    roughness_pred = roughness_pred.detach().cpu().numpy().transpose(0,2,3,1)[0]*255
    
    base_color_img = np.concatenate([render_view_img[..., ::-1], base_color_pred[..., ::-1], base_color_img[..., ::-1]], axis=1)
    normal_img = np.concatenate([render_view_img[..., ::-1], normal_pred[..., ::-1], normal_img[..., ::-1]], axis=1)
    metallic_img = np.concatenate([render_view_img[..., ::-1], cv2.cvtColor(metallic_pred, cv2.COLOR_GRAY2BGR), metallic_img[..., ::-1]], axis=1)
    roughness_img = np.concatenate([render_view_img[..., ::-1], cv2.cvtColor(roughness_pred, cv2.COLOR_GRAY2BGR), roughness_img[..., ::-1]], axis=1)
    
    img = np.concatenate([base_color_img, normal_img, metallic_img, roughness_img], axis=0)
    cv2.imwrite('test.jpg', img)

def train(vis=False):
    # Set seed
    seed_everything(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set hyperparameters
    batch_size = 4
    num_epochs = 100
    learning_rate = 0.001
    
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device).reshape(1,3,1,1)
    
    # Set dataset
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((512, 512)),
    ])
    label_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
    ])
    train_dataset = SVABOMaterialDataset(root_dir='../../datasets/abo-benchmark-material', image_transform=image_transform, label_transform=label_transform, train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    # Set model
    model = SVNet(activation='relu').to(device)
    
    # Set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Set TensorBoard
    # writer = tensorboard.SummaryWriter()
    
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (render_view_img, base_color_img, normal_img, metallic_roughness_img, mask_img) in enumerate(train_loader):
            optimizer.zero_grad()
            render_view_img = render_view_img.to(device)
            base_color_img = base_color_img.to(device)
            normal_img = normal_img.to(device)
            metallic_roughness_img = metallic_roughness_img.to(device)
            mask_img = mask_img.to(device)
            
            base_color_img = base_color_img*mask_img
            
            # Forward pass
            base_color_pred, normal_pred, metallic_pred, roughness_pred = model(render_view_img)
            base_color_pred = F.interpolate(base_color_pred, size=(512, 512), mode='bilinear', align_corners=False)
            normal_pred = F.interpolate(normal_pred, size=(512, 512), mode='bilinear', align_corners=False)
            metallic_pred = F.interpolate(metallic_pred, size=(512, 512), mode='bilinear', align_corners=False)
            roughness_pred = F.interpolate(roughness_pred, size=(512, 512), mode='bilinear', align_corners=False)

            # Compute loss
            base_color_loss = criterion(base_color_pred, base_color_img)
            normal_loss = criterion(normal_pred, normal_img)
            metallic_loss = criterion(metallic_pred, metallic_roughness_img[:, 0:1])
            roughness_loss = criterion(roughness_pred, metallic_roughness_img[:, 1:2])
            
            loss = (base_color_loss + normal_loss + metallic_loss + roughness_loss)/4
            loss.backward()
            optimizer.step()
            
            # Visualization
            if vis:
                visualize(render_view_img, base_color_img, normal_img, metallic_roughness_img,
                        mask_img, base_color_pred, normal_pred, metallic_pred, roughness_pred,
                        mean, std)

train(True)