import os

import numpy as np
import cv2

def load_texture(file_path):
    texture = cv2.imread(file_path, cv2.IMREAD_COLOR)
    texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
    texture = texture / 255.0  # Normalize to [0, 1]
    return texture

def load_metallic_roughness(file_path):
    metallic_roughness = cv2.imread(file_path, cv2.IMREAD_COLOR)
    metallic = metallic_roughness[..., 0:1]
    roughness = metallic_roughness[..., 1:2]
    return metallic / 255.0, roughness / 255.0  # Normalize to [0, 1]

def load_normal(file_path):
    normal = cv2.imread(file_path, cv2.IMREAD_COLOR)
    normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
    normal = (normal / 255.0) * 2.0 - 1.0  # Normalize to [-1, 1]
    return normal

def render(base_color, metallic, roughness, normal, light_dir, view_dir):
    height, width, _ = base_color.shape
    result = np.zeros((height, width, 3))
    
    light_dir = light_dir / np.linalg.norm(light_dir)
    view_dir = view_dir / np.linalg.norm(view_dir)
    
    for y in range(height):
        for x in range(width):
            n = normal[y, x]
            n = n / np.linalg.norm(n)
            
            l = light_dir
            v = view_dir
            
            h = (l + v) / np.linalg.norm(l + v)
            
            NdotL = max(np.dot(n, l), 0.0)
            NdotV = max(np.dot(n, v), 0.0)
            NdotH = max(np.dot(n, h), 0.0)
            VdotH = max(np.dot(v, h), 0.0)
            
            base = base_color[y, x]
            metal = metallic[y, x]
            rough = roughness[y, x]
            
            F0 = np.array([0.04, 0.04, 0.04]) * (1 - metal) + base * metal
            
            F = F0 + (1 - F0) * (1 - VdotH) ** 5
            D = (rough ** 2) / (np.pi * ((NdotH ** 2) * (rough ** 2 - 1) + 1) ** 2)
            k = (rough + 1) ** 2 / 8
            G = NdotL * NdotV / (NdotL * (1 - k) + k) / (NdotV * (1 - k) + k)
            
            specular = F * D * G / (4 * NdotL * NdotV + 1e-5)
            diffuse = (1 - F) * base / np.pi
            
            result[y, x] = NdotL * (diffuse + specular)
    
    return np.clip(result, 0, 1)


def load_environment_map(file_path):
    env_map = cv2.imread(file_path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    env_map = cv2.cvtColor(env_map, cv2.COLOR_BGR2RGB)
    return env_map

def sample_environment_map(env_map, direction):
    height, width, _ = env_map.shape
    u = 0.5 + (np.arctan2(direction[2], direction[0]) / (2 * np.pi))
    v = 0.5 - (np.arcsin(direction[1]) / np.pi)
    x = int(u * width) % width
    y = int(v * height) % height
    return env_map[y, x]

def render_with_environment_map(base_color, metallic, roughness, normal, env_map, view_dir):
    height, width, _ = base_color.shape
    result = np.zeros((height, width, 3))

    view_dir = view_dir / np.linalg.norm(view_dir)
    
    # Hemisphere sampling
    num_samples = 64
    samples = np.random.randn(num_samples, 3)
    samples = samples / np.linalg.norm(samples, axis=1)[:, np.newaxis]
    samples = samples[samples[:, 1] > 0]  # Keep only upper hemisphere
    
    for y in range(height):
        for x in range(width):
            n = normal[y, x]
            n = n / np.linalg.norm(n)
            
            v = view_dir
            base = base_color[y, x]
            metal = metallic[y, x]
            rough = roughness[y, x]

            F0 = np.array([0.04, 0.04, 0.04]) * (1 - metal) + base * metal
            total_light = np.zeros(3)

            for sample in samples:
                l = sample / np.linalg.norm(sample)
                h = (v + l) / np.linalg.norm(v + l)
                
                NdotL = max(np.dot(n, l), 0.0)
                NdotV = max(np.dot(n, v), 0.0)
                NdotH = max(np.dot(n, h), 0.0)
                VdotH = max(np.dot(v, h), 0.0)
                
                F = F0 + (1 - F0) * (1 - VdotH) ** 5
                D = (rough ** 2) / (np.pi * ((NdotH ** 2) * (rough ** 2 - 1) + 1) ** 2)
                k = (rough + 1) ** 2 / 8
                G = NdotL * NdotV / (NdotL * (1 - k) + k) / (NdotV * (1 - k) + k)
                
                specular = F * D * G / (4 * NdotL * NdotV + 1e-5)
                diffuse = (1 - F) * base / np.pi
                
                env_light = sample_environment_map(env_map, l)
                total_light += NdotL * env_light * (diffuse + specular)
            
            result[y, x] = total_light / len(samples)
    
    return np.clip(result, 0, 1)

if __name__ == '__main__':
    root_dir = os.path.join('../..', 'datasets', 'abo-benchmark-material')
    
    for model_id in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, model_id)):
            break
    
    for i in range(91):
        # 파일 경로를 실제 텍스처 파일로 변경하세요
        base_color = load_texture(os.path.join(root_dir, model_id, 'base_color', f'base_color_{i}.jpg'))
        normal = load_normal(os.path.join(root_dir, model_id, 'normal', f'normal_{i}.png'))
        metallic, roughness = load_metallic_roughness(os.path.join(root_dir, model_id, 
                                                                   'metallic_roughness', f'metallic_roughness_{i}.jpg'))
        
        render_path = os.path.join(root_dir, model_id, 'render', '0', f'render_{i}.jpg')
        os.system(f'cp {render_path} {'./render_view.jpg'}')
        cv2.imwrite('base_color.png', (base_color * 255).astype(np.uint8))
        cv2.imwrite('normal.png', ((normal + 1) * 127.5).astype(np.uint8))
        cv2.imwrite('metallic.png', (metallic * 255).astype(np.uint8))

        light_dir = np.array([0, 0, 1])  # 예: z 방향으로 오는 빛
        view_dir = np.array([0, 0, 1])  # 예: z 방향에서 보는 시점

        result = render(base_color, metallic, roughness, normal, light_dir, view_dir)

        # 결과 이미지를 저장합니다
        cv2.imwrite('rendered_image.png', (result * 255).astype(np.uint8))

