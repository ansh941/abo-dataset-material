import os

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm

import tensorboardX

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
    base_color = np.concatenate([render_view, base_color_pred, base_color], axis=1)
    normal = np.concatenate([render_view, normal_pred, normal], axis=1)
    metallic = np.concatenate([render_view, cv2.cvtColor(metallic_pred, cv2.COLOR_GRAY2BGR), metallic], axis=1)
    roughness = np.concatenate([render_view, cv2.cvtColor(roughness_pred, cv2.COLOR_GRAY2BGR), roughness], axis=1)
    recon_view = np.concatenate([render_view, recon_view_pred, recon_view], axis=1)
    
    img = np.concatenate([base_color, normal, metallic, roughness, recon_view], axis=0)
    cv2.imwrite('test.jpg', img)


def rmse_loss_with_mask(pred, target, criterion, mask):
    return torch.sum(torch.sqrt(criterion(pred, target) * mask)) / torch.count_nonzero(mask)

def train(vis=False):
    # Set seed
    seed_everything(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set hyperparameters
    batch_size = 4
    num_epochs = 20
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
    
    test_dataset = SVABOMaterialDataset(root_dir='../../datasets/abo-benchmark-material', image_transform=image_transform, label_transform=label_transform, train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # Set model
    model = SVNet(activation='relu').to(device)
    
    # Set loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Set TensorBoard
    writer = tensorboardX.SummaryWriter()
    
    light_dir = torch.tensor([0,0,1]).to(device).float()
    view_dir = torch.tensor([0,0,1]).to(device).float()
    
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm.tqdm(range(len(train_loader)))
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
            
            size = base_color_pred.shape[-2:]
            render_view = F.interpolate(render_view, size=size, mode='bilinear', align_corners=False)
            base_color = F.interpolate(base_color, size=size, mode='bilinear', align_corners=False)
            normal = F.interpolate(normal, size=size, mode='bilinear', align_corners=False)
            metallic = F.interpolate(metallic, size=size, mode='bilinear', align_corners=False)
            roughness = F.interpolate(roughness, size=size, mode='bilinear', align_corners=False)
            mask = F.interpolate(mask, size=size, mode='bilinear', align_corners=False)
            recon_view = F.interpolate(recon_view, size=size, mode='bilinear', align_corners=False)

            # Rendering
            recon_view_pred = torch.zeros((base_color_pred.shape[0], size[0], size[1], 3), device=device).float()
            for idx in range(len(base_color_pred)):
                recon_view_pred[idx] = render_torch(base_color_pred, metallic_pred, roughness_pred, normal_pred, light_dir, view_dir, idx)
            recon_view_pred = recon_view_pred.permute(0, 3, 1, 2)

            # Compute loss
            mask = torch.where(mask > 0.5, mask, 0)
            nonzeros = torch.count_nonzero(mask)
            base_color_loss = rmse_loss_with_mask(base_color_pred, base_color, criterion, mask)
            normal_loss = rmse_loss_with_mask(normal_pred, normal, criterion, mask)
            metallic_loss = rmse_loss_with_mask(metallic_pred, metallic, criterion, mask)
            roughness_loss = rmse_loss_with_mask(roughness_pred, roughness, criterion, mask)
            rendering_loss = rmse_loss_with_mask(recon_view_pred, recon_view, criterion, mask)
            loss = (base_color_loss + normal_loss + metallic_loss + roughness_loss + rendering_loss)/5
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if i % 100 == 0:
                print('Train - Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                # Visualization
                if vis:
                    visualize(render_view, base_color, normal, metallic, roughness, mask, recon_view,
                            base_color_pred, normal_pred, metallic_pred, roughness_pred, recon_view_pred,
                            mean, std)
            progress_bar.update(len(render_view))
        total_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/train', total_loss, epoch)
        
        # Evaluation
        model.eval()
        val_losses = []
        with torch.no_grad():
            val_loss = 0
            progress_bar = tqdm.tqdm(range(len(test_loader)))
            for i, (render_view, base_color, normal, metallic, roughness, mask, recon_view) in enumerate(test_loader):
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
                base_color_loss = torch.sqrt(criterion(base_color_pred, base_color))
                normal_loss = torch.sqrt(criterion(normal_pred, normal))
                metallic_loss = torch.sqrt(criterion(metallic_pred, metallic))
                roughness_loss = torch.sqrt(criterion(roughness_pred, roughness))
                rendering_loss = torch.sqrt(criterion(recon_view_pred, recon_view))
                loss = (base_color_loss + normal_loss + metallic_loss + roughness_loss + rendering_loss)/5
                
                val_loss += loss.item()
                progress_bar.update(len(render_view))
                
            val_loss = val_loss / len(test_loader)
            print('Test - Epoch [{}/{}] Test Loss: {:.4f}'.format(epoch+1, num_epochs, val_loss))
            writer.add_scalar('Loss/test', val_loss, epoch)
            
            if val_losses[-1] > val_loss:
                torch.save(model.state_dict(), f'model_{epoch+1}_{val_loss}.pth')
            val_losses.append(val_loss)

train(True)