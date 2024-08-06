import torch
import torch.nn as nn

def rmse_loss_with_mask(pred, target, criterion, mask):
    criterion = nn.MSELoss(reduction='mean')
    return torch.sqrt(criterion(pred*mask, target*mask))
