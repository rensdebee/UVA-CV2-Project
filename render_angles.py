import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam

def render_images(renderer, cam, obj_path, options):
    images = []
    poses = []
    vers, hors, radii = [], [], []
    
    min_ver = max(min(options['min_ver'], options['min_ver'] - options['elevation']), -80 - options['elevation'])
    max_ver = min(max(options['max_ver'], options['max_ver'] - options['elevation']), 80 - options['elevation'])

    for _ in range(options['batch_size']):
        ver = np.random.randint(min_ver, max_ver)
        hor = np.random.randint(-180, 180)
        radius = 0

        vers.append(ver)
        hors.append(hor)
        radii.append(radius)

        pose = orbit_camera(options['elevation'] + ver, hor, options['radius'] + radius)
        poses.append(pose)

        cur_cam = MiniCam(pose, options['render_resolution'], options['render_resolution'], cam.fovy, cam.fovx, cam.near, cam.far)
        
        bg_color = torch.tensor([1, 1, 1] if np.random.rand() > options['invert_bg_prob'] else [0, 0, 0], dtype=torch.float32, device="cuda")
        out = renderer.render(cur_cam, obj_path=obj_path, bg_color=bg_color)

        image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
        images.append(image)
    
    return images, poses


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate CLIP similarity for rendered images from different angles and average the scores.")
    parser.add_argument("directory", type=str, help="Path to the directory to save rendered images.")
    parser.add_argument("obj_path", type=str, help="Path to the .obj file to render.")
    args = parser.parse_args()

    renderer = Renderer()
    cam = MiniCam()
    options = {
            'min_ver': -30,
            'max_ver': 30,
            'elevation': 0,
            'radius': 10,
            'render_resolution': 256,
            'invert_bg_prob': 0.1,
            'batch_size': 10
        }
    
    images, _ = render_images(renderer, cam, args.obj_path, options)

    for idx, image_tensor in enumerate(images):
        image_path = os.path.join(args.directory, f'rendered_{idx}.png')
        image = Image.fromarray((image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        image.save(image_path)