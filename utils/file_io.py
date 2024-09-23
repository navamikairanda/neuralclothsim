import os
import imageio
import torch
from pytorch3d.io import save_obj
from utils.config_parser import device 

def save_meshes(positions, faces, meshes_dir, i, verts_uvs=None, tex_image_file=None):
    meshes_dir = os.path.join(meshes_dir, f'{i}')
    os.makedirs(meshes_dir, exist_ok=True)
    if tex_image_file is not None:
        texture_map = torch.tensor(imageio.imread(tex_image_file)/255., dtype=torch.float, device=device)
    else:
        texture_map = None
    save_obj(os.path.join(meshes_dir, 'simulated.obj'), positions[0], faces, verts_uvs=verts_uvs, faces_uvs=faces, texture_map=texture_map)