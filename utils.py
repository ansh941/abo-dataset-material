import numpy as np
import cv2
import torch
import torch.nn.functional as F

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def gpu_to_cpu_for_visualizing(base_color, normal, metallic, roughness, mask, recon_view, index):
    base_color = base_color.detach().cpu().numpy().transpose(0,2,3,1)[index]
    normal = normal.detach().cpu().numpy().transpose(0,2,3,1)[index]
    metallic = metallic.detach().cpu().numpy().transpose(0,2,3,1)[index]
    roughness = roughness.detach().cpu().numpy().transpose(0,2,3,1)[index]
    recon_view = recon_view.detach().cpu().numpy().transpose(0,2,3,1)[index]
    
    if mask is None:
        return base_color, normal, metallic, roughness, recon_view
    
    mask = mask.detach().cpu().numpy().transpose(0,2,3,1)[index]
    return base_color, normal, metallic, roughness, mask, recon_view

def only_object(base_color, normal, metallic, roughness, recon_view, mask):
    base_color = base_color * mask
    normal = normal * mask
    metallic = metallic * mask
    roughness = roughness * mask
    recon_view = recon_view * mask
    
    return base_color, normal, metallic, roughness, recon_view

def element_concat(base_color, normal, metallic, roughness, recon_view):
    normal = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
    if metallic.shape[-1] == 1:
        metallic = cv2.cvtColor(metallic, cv2.COLOR_GRAY2BGR)
    if roughness.shape[-1] == 1:
        roughness = cv2.cvtColor(roughness, cv2.COLOR_GRAY2BGR)
    return np.vstack([base_color, normal, metallic, roughness, recon_view])

def visualize(render_view, 
              base_color_pred, normal_pred, metallic_pred, roughness_pred, recon_view_pred,
              mean, std, index = 0, with_gt = False, 
              base_color = None, normal = None, metallic = None, roughness = None, mask = None, recon_view = None,
              ):
    image_list = []
    
    render_view = (render_view * std + mean).detach().cpu().numpy().transpose(0,2,3,1)[index]
    
    base_color_pred, normal_pred, metallic_pred, roughness_pred, recon_view_pred = gpu_to_cpu_for_visualizing(base_color_pred, normal_pred, metallic_pred, roughness_pred, None, recon_view_pred, index)
    
    black_space = np.zeros_like(base_color_pred)
    
    left = element_concat(black_space, black_space, render_view, black_space, black_space)
    image_list.append(left)
    
    if with_gt:
        base_color, normal, metallic, roughness, mask, recon_view = gpu_to_cpu_for_visualizing(base_color, normal, metallic, roughness, mask, recon_view, index)
        
        base_color_pred, normal_pred, metallic_pred, roughness_pred, recon_view_pred = only_object(base_color_pred, normal_pred, metallic_pred, roughness_pred, recon_view_pred, mask)
        base_color, normal, metallic, roughness, recon_view = only_object(base_color, normal, metallic, roughness, recon_view, mask)
        
        center = element_concat(base_color, normal, metallic, roughness, recon_view)
        image_list.append(center)
        
    right = element_concat(base_color_pred, normal_pred, metallic_pred, roughness_pred, recon_view_pred)
    image_list.append(right)
    
    img = np.hstack(image_list)*255
    cv2.imwrite('test.jpg', img)

def align_size(size, render_view, base_color=None, normal=None, metallic=None, roughness=None, mask=None, recon_view=None):
    render_view = F.interpolate(render_view, size=size, mode='bilinear', align_corners=False)
    if base_color is None:
        return render_view
    
    base_color = F.interpolate(base_color, size=size, mode='bilinear', align_corners=False)
    normal = F.interpolate(normal, size=size, mode='bilinear', align_corners=False)
    metallic = F.interpolate(metallic, size=size, mode='bilinear', align_corners=False)
    roughness = F.interpolate(roughness, size=size, mode='bilinear', align_corners=False)
    mask = F.interpolate(mask, size=size, mode='bilinear', align_corners=False)
    recon_view = F.interpolate(recon_view, size=size, mode='bilinear', align_corners=False)
    
    return render_view, base_color, normal, metallic, roughness, mask, recon_view