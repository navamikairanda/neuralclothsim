import os
import imageio
import torch
from pytorch3d.io import save_obj
from config_parser import device 

def save_meshes(positions, faces, meshes_dir, i, temporal_sidelen, verts_uvs=None, tex_image_file=None):
    n_vertices = positions.shape[1] // temporal_sidelen
    meshes_dir = os.path.join(meshes_dir, f'{i}')
    os.makedirs(meshes_dir, exist_ok=True)
    if tex_image_file is not None:
        texture_map = torch.tensor(imageio.imread(tex_image_file)/255., dtype=torch.float, device=device)
    else:
        texture_map = None
    for surface_idx in range(temporal_sidelen):
        verts = positions[:,surface_idx * n_vertices:(surface_idx + 1) * n_vertices][0] 
        save_obj(os.path.join(meshes_dir, f'{surface_idx:03d}.obj'), verts, faces, verts_uvs=verts_uvs, faces_uvs=faces, texture_map=texture_map)