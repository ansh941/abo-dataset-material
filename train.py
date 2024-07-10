import os

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm

import tensorboard

from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.abo import SVABOMaterialDataset
from models.svnet import SVNet

from rendering_test import render_torch

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def visualize(render_view, base_color, normal, metallic, roughness, mask, recon_view,
              base_color_pred, normal_pred, metallic_pred, roughness_pred, recon_view_pred,
              mean, std):
    render_view = (render_view * std + mean).detach().cpu().numpy().transpose(0,2,3,1)[0]*255
    base_color = base_color.detach().cpu().numpy().transpose(0,2,3,1)[0]*255
    normal = normal.detach().cpu().numpy().transpose(0,2,3,1)[0]*255
    metallic = cv2.cvtColor(metallic.detach().cpu().numpy().transpose(0,2,3,1)[0]*255, cv2.COLOR_GRAY2BGR)
    roughness = cv2.cvtColor(roughness.detach().cpu().numpy().transpose(0,2,3,1)[0]*255, cv2.COLOR_GRAY2BGR)
    recon_view = recon_view.detach().cpu().numpy().transpose(0,2,3,1)[0]*255
    
    base_color_pred = base_color_pred.detach().cpu().numpy().transpose(0,2,3,1)[0]*255
    normal_pred = normal_pred.detach().cpu().numpy().transpose(0,2,3,1)[0]*255
    metallic_pred = metallic_pred.detach().cpu().numpy().transpose(0,2,3,1)[0]*255
    roughness_pred = roughness_pred.detach().cpu().numpy().transpose(0,2,3,1)[0]*255
    recon_view_pred = recon_view_pred.detach().cpu().numpy().transpose(0,2,3,1)[0]*255
    
    render_view = render_view[..., ::-1]
    base_color = np.concatenate([render_view, base_color_pred[..., ::-1], base_color[..., ::-1]], axis=1)
    normal = np.concatenate([render_view, normal_pred[..., ::-1], normal[..., ::-1]], axis=1)
    metallic = np.concatenate([render_view, cv2.cvtColor(metallic_pred, cv2.COLOR_GRAY2BGR), metallic[..., ::-1]], axis=1)
    roughness = np.concatenate([render_view, cv2.cvtColor(roughness_pred, cv2.COLOR_GRAY2BGR), roughness[..., ::-1]], axis=1)
    recon_view = np.concatenate([render_view, recon_view_pred[..., ::-1], recon_view[..., ::-1]], axis=1)
    
    img = np.concatenate([base_color, normal, metallic, roughness, recon_view], axis=0)
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
    
    light_dir = torch.tensor([0,0,1]).to(device).float()
    view_dir = torch.tensor([0,0,1]).to(device).float()
    
    # Train the model
    total_step = len(train_loader)
    for epoch in tqdm.tqdm(range(num_epochs)):
        for i, (render_view, base_color, normal, metallic, roughness, mask, recon_view) in enumerate(train_loader):
            optimizer.zero_grad()
            render_view = render_view.to(device).float()
            base_color = base_color.to(device).float()
            normal = normal.to(device).float()
            metallic = metallic.to(device).float()
            roughness = roughness.to(device).float()
            mask = mask.to(device).float()
            recon_view = recon_view.to(device).float()
            
            base_color = base_color*mask
            normal = normal*mask
            metallic = metallic*mask
            roughness = roughness*mask
            recon_view = recon_view*mask
            
            # Forward pass
            base_color_pred, normal_pred, metallic_pred, roughness_pred = model(render_view)
            base_color_pred = F.interpolate(base_color_pred, size=(512, 512), mode='bilinear', align_corners=False)
            normal_pred = F.interpolate(normal_pred, size=(512, 512), mode='bilinear', align_corners=False)
            metallic_pred = F.interpolate(metallic_pred, size=(512, 512), mode='bilinear', align_corners=False)
            roughness_pred = F.interpolate(roughness_pred, size=(512, 512), mode='bilinear', align_corners=False)

            # Rendering
            recon_view_pred = torch.zeros((base_color_pred.shape[0], 512, 512, 3), device=device).float()
            for idx in range(len(base_color_pred)):
                recon_view_pred[idx] = render_torch(base_color_pred, metallic_pred, roughness_pred, normal_pred, light_dir, view_dir, idx)
            recon_view_pred = recon_view_pred.permute(0, 3, 1, 2)
            
            # Compute loss
            base_color_loss = criterion(base_color_pred, base_color)
            normal_loss = criterion(normal_pred, normal)
            metallic_loss = criterion(metallic_pred, metallic)
            roughness_loss = criterion(roughness_pred, roughness)
            rendering_loss = criterion(recon_view_pred, recon_view)
            loss = (base_color_loss + normal_loss + metallic_loss + roughness_loss + rendering_loss)/5
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
            # Visualization
            if vis:
                visualize(render_view, base_color, normal, metallic, roughness, mask, recon_view,
                        base_color_pred, normal_pred, metallic_pred, roughness_pred, recon_view_pred,
                        mean, std)

train(True)