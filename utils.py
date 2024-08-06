import numpy as np
import cv2
import torch

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
