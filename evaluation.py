import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm

from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.abo import SVABOMaterialDataset
from models.svnet import SVNet

from utils import visualize, seed_everything
from loss import rmse_loss_with_mask
from rendering_test import render_torch

def evaluation(model_path='pretrained/svnet.pth'):
    seed_everything(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set hyperparameters
    batch_size = 16
    
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
    
    test_dataset = SVABOMaterialDataset(root_dir='../../datasets/abo-benchmark-material', image_transform=image_transform, label_transform=label_transform, train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # Set model
    model = SVNet(activation='relu').to(device)
    model.load_state_dict(torch.load(model_path))
    
    criterion = nn.MSELoss(reduction='none')
    
    light_dir = torch.tensor([0,0,1]).to(device).float()
    view_dir = torch.tensor([0,0,1]).to(device).float()
    
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        progress_bar = tqdm.tqdm(range(len(test_dataset)))
        for i, (render_view, base_color, normal, metallic, roughness, mask, recon_view) in enumerate(test_loader):
            render_view = render_view.to(device).float()
            base_color = base_color.to(device).float()
            normal = normal.to(device).float()
            metallic = metallic.to(device).float()
            roughness = roughness.to(device).float()
            mask = mask.to(device).float()
            recon_view = recon_view.to(device).float()
            
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
            mask = torch.where(mask > 0.5, mask, 1e-6)
            base_color_loss = rmse_loss_with_mask(base_color_pred, base_color, criterion, mask)
            normal_loss = rmse_loss_with_mask(normal_pred, normal, criterion, mask)
            metallic_loss = rmse_loss_with_mask(metallic_pred, metallic, criterion, mask)
            roughness_loss = rmse_loss_with_mask(roughness_pred, roughness, criterion, mask)
            rendering_loss = rmse_loss_with_mask(recon_view_pred, recon_view, criterion, mask)
            loss = (base_color_loss + normal_loss + metallic_loss + roughness_loss + rendering_loss)/5
            
            val_loss += loss.item()
            progress_bar.update(len(render_view))
            
        val_loss = val_loss / len(test_loader)
        print('Test Loss: {:.4f}'.format(val_loss))
        
        visualize(render_view, base_color, normal, metallic, roughness, mask, recon_view,
                            base_color_pred, normal_pred, metallic_pred, roughness_pred, recon_view_pred,
                            mean, std)

if __name__ == '__main__':
    evaluation(model_path='pretrained/svnet.pth')