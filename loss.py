import torch
import torch.nn as nn
import torch.nn.functional as F

def rmse_loss_with_mask(pred, target, criterion, mask):
    criterion = nn.MSELoss(reduction='mean')
    return torch.sqrt(criterion(pred*mask, target*mask))

def cosine_similarity_with_mask(pred:torch.tensor, target:torch.tensor, mask:torch.tensor):
    pred = pred * mask
    target = target * mask
    pred = pred.view(pred.shape[0], -1)
    target = target.view(target.shape[0], -1)
    pred = F.normalize(pred, p=2, dim=1)
    target = F.normalize(target, p=2, dim=1)
    similarity = (pred * target).sum(dim=1)
    return similarity.mean()

def compute_loss(base_color, normal, metallic, roughness, recon_view, 
                 base_color_pred, normal_pred, metallic_pred, roughness_pred, recon_view_pred,
                 mask, criterion):
    mask = torch.where(mask > 0.5, mask, 1e-6)
    base_color_loss = rmse_loss_with_mask(base_color_pred, base_color, criterion, mask)
    normal_loss = rmse_loss_with_mask(normal_pred, normal, criterion, mask)
    metallic_loss = rmse_loss_with_mask(metallic_pred, metallic, criterion, mask)
    roughness_loss = rmse_loss_with_mask(roughness_pred, roughness, criterion, mask)
    rendering_loss = rmse_loss_with_mask(recon_view_pred, recon_view, criterion, mask)
    loss = (base_color_loss + normal_loss + metallic_loss + roughness_loss + rendering_loss)/5

    normal_cos = cosine_similarity_with_mask(normal_pred, normal, mask)
    
    return (loss, base_color_loss, normal_loss, metallic_loss, 
            roughness_loss, rendering_loss, normal_cos)
