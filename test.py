import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from models.svnet import SVNet
from utils import seed_everything, align_size
from rendering_test import render_torch
from utils import visualize

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
        render_view = align_size(size, render_view)

        # Rendering
        recon_view_pred = torch.zeros((base_color_pred.shape[0], size[0], size[1], 3), device=device).float()
        for idx in range(len(base_color_pred)):
            recon_view_pred[idx] = render_torch(base_color_pred, metallic_pred, roughness_pred, normal_pred, light_dir, view_dir, idx)
        recon_view_pred = recon_view_pred.permute(0, 3, 1, 2)

        # visualize
        visualize_fn_args = {
            'render_view': render_view,
            'base_color_pred': base_color_pred,
            'normal_pred': normal_pred,
            'metallic_pred': metallic_pred,
            'roughness_pred': roughness_pred,
            'recon_view_pred': recon_view_pred,
            'mean': mean,
            'std': std,
            'index': 0,
            'with_gt': False,
        }
        visualize(**visualize_fn_args)

if __name__ == '__main__':
    test(image_path='/home/ash/Desktop/datasets/abo-benchmark-material/B07MF1SFKD/render/0/render_0.jpg', model_path='pretrained/svnet.pth')