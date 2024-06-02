import os
import cv2
import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt

from mesh import Mesh, safe_normalize
from mesh_renderer import Renderer

class Options:
    def __init__(self):
        self.mesh = '/home/scur2155/CV2/UVA-CV2-Project/sugar_output/results_car/extracted_mesh.obj'
        self.texture_lr = 0.001
        self.geom_lr = 0.001
        self.train_geo = True
        self.force_cuda_rast = False
        self.gui = False

def load_texture(path):
    texture = plt.imread(path)
    texture = (texture * 255).astype(np.uint8)
    return torch.tensor(texture).float() / 255

def main():
    opt = Options()
    
    # Load the texture
    texture_path = '/home/scur2155/CV2/UVA-CV2-Project/sugar_output/results_car/extracted_mesh.png'
    texture = load_texture(texture_path)

    # Load the mesh
    mesh = Mesh.load(opt.mesh, resize=False)
    mesh.albedo = texture

    # Initialize the renderer
    renderer = Renderer(opt)
    renderer.mesh = mesh

    # Define a simple camera pose and projection
    pose = np.eye(4, dtype=np.float32)
    proj = np.eye(4, dtype=np.float32)

    # Render the mesh
    result = renderer.render(pose, proj, 800, 800, ssaa=1)

    # Display the rendered image
    rendered_image = result['image'].cpu().numpy()
    plt.imshow(rendered_image)
    plt.show()

if __name__ == '__main__':
    main()
