import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm

from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.abo import SVABOMaterialDataset
from models.svnet import SVNet

from utils import visualize, seed_everything, align_size
from loss import compute_loss
from rendering import render_torch


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
        base_color_rmse = 0
        metallic_rmse = 0
        roughness_rmse = 0
        rendering_rmse = 0
        normal_cosine = 0
        
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
            render_view, base_color, normal, metallic, roughness, mask, recon_view = align_size(size, render_view, base_color, normal, metallic, roughness, mask, recon_view)

            # Rendering
            recon_view_pred = torch.zeros((base_color_pred.shape[0], size[0], size[1], 3), device=device).float()
            for idx in range(len(base_color_pred)):
                recon_view_pred[idx] = render_torch(base_color_pred, metallic_pred, roughness_pred, normal_pred, light_dir, view_dir, idx)
            recon_view_pred = recon_view_pred.permute(0, 3, 1, 2)

            # Compute loss
            loss_fn_args = {
                'base_color': base_color,
                'normal': normal,
                'metallic': metallic,
                'roughness': roughness,
                'recon_view': recon_view,
                'base_color_pred': base_color_pred,
                'normal_pred': normal_pred,
                'metallic_pred': metallic_pred,
                'roughness_pred': roughness_pred,
                'recon_view_pred': recon_view_pred,
                'mask': mask,
                'criterion': criterion
            }
            (loss, base_color_loss, normal_loss, metallic_loss, roughness_loss, rendering_loss, normal_cos) = compute_loss(**loss_fn_args)
            
            val_loss += loss.item()
            base_color_rmse += base_color_loss.item()
            metallic_rmse += metallic_loss.item()
            roughness_rmse += roughness_loss.item()
            rendering_rmse += rendering_loss.item()
            normal_cosine += normal_cos.item()
            
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
                'with_gt': True,
                'base_color': base_color,
                'normal': normal,
                'metallic': metallic,
                'roughness': roughness,
                'mask': mask,
                'recon_view': recon_view
            }
            visualize(**visualize_fn_args)
        
            progress_bar.update(len(render_view))
            
        val_loss = val_loss / len(test_loader)
        base_color_rmse = base_color_rmse / len(test_loader)
        metallic_rmse = metallic_rmse / len(test_loader)
        roughness_rmse = roughness_rmse / len(test_loader)
        rendering_rmse = rendering_rmse / len(test_loader)
        normal_cosine = normal_cosine / len(test_loader)
        
        print('Test Loss: {:.4f}'.format(val_loss))
        print('Base Color RMSE: {:.4f}'.format(base_color_rmse))
        print('Metallic RMSE: {:.4f}'.format(metallic_rmse))
        print('Roughness RMSE: {:.4f}'.format(roughness_rmse))
        print('Rendering RMSE: {:.4f}'.format(rendering_rmse))
        print('Normal Cosine Similarity: {:.4f}'.format(normal_cosine))
        
        return val_loss
        
if __name__ == '__main__':
    evaluation(model_path='pretrained/svnet.pth')