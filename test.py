import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from models.svnet import SVNet
from utils import seed_everything
from rendering_test import render_torch

def test(image_path, model_path='pretrained/svnet.pth'):
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
    # Set model
    model = SVNet(activation='relu').to(device)
    model.load_state_dict(torch.load(model_path))
    
    
    light_dir = torch.tensor([0,0,1]).to(device).float()
    view_dir = torch.tensor([0,0,1]).to(device).float()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        render_view = cv2.imread(image_path, 1) / 255
        
        render_view = image_transform(render_view)

        # Forward pass
        base_color_pred, normal_pred, metallic_pred, roughness_pred = model(render_view)
        
        size = base_color_pred.shape[-2:]
        render_view = F.interpolate(render_view, size=size, mode='bilinear', align_corners=False)
        base_color = F.interpolate(base_color, size=size, mode='bilinear', align_corners=False)
        normal = F.interpolate(normal, size=size, mode='bilinear', align_corners=False)
        metallic = F.interpolate(metallic, size=size, mode='bilinear', align_corners=False)
        roughness = F.interpolate(roughness, size=size, mode='bilinear', align_corners=False)
        recon_view = F.interpolate(recon_view, size=size, mode='bilinear', align_corners=False)

        # visualize
        render_view = (render_view * std + mean).detach().cpu().numpy().transpose(0,2,3,1)[0]*255
        
        base_color_pred = base_color_pred.detach().cpu().numpy().transpose(0,2,3,1)[0]*255
        normal_pred = normal_pred.detach().cpu().numpy().transpose(0,2,3,1)[0]*255
        metallic_pred = metallic_pred.detach().cpu().numpy().transpose(0,2,3,1)[0]*255
        roughness_pred = roughness_pred.detach().cpu().numpy().transpose(0,2,3,1)[0]*255
        recon_view_pred = recon_view_pred.detach().cpu().numpy().transpose(0,2,3,1)[0]*255
        
        # Rendering
        recon_view_pred = torch.zeros((base_color_pred.shape[0], size[0], size[1], 3), device=device).float()
        for idx in range(len(base_color_pred)):
            recon_view_pred[idx] = render_torch(base_color_pred, metallic_pred, roughness_pred, normal_pred, light_dir, view_dir, idx)
        recon_view_pred = recon_view_pred.permute(0, 3, 1, 2)
        
        render_view = render_view[..., ::-1]
        base_color = np.concatenate([render_view, base_color_pred], axis=1)
        normal = np.concatenate([render_view, normal_pred], axis=1)
        metallic = np.concatenate([render_view, cv2.cvtColor(metallic_pred, cv2.COLOR_GRAY2BGR)], axis=1)
        roughness = np.concatenate([render_view, cv2.cvtColor(roughness_pred, cv2.COLOR_GRAY2BGR)], axis=1)
        recon_view = np.concatenate([render_view, recon_view_pred], axis=1)
        
        img = np.concatenate([base_color, normal, metallic, roughness, recon_view], axis=0)
        
        cv2.imwrite('test.jpg', img)

if __name__ == '__main__':
    test(model_path='pretrained/svnet.pth')