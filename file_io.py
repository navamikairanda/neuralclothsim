import os
import numpy as np
import imageio
import natsort
import torch
from pytorch3d.io import save_obj, load_obj
from pytorch3d.structures import Pointclouds
from pytorch3d.structures import Meshes, join_meshes_as_batch
from config_parser import device

to8b = lambda x : (255 * np.clip(x, 0, 1)).astype(np.uint8)

def save_images(images, images_dir, i, save_video=True, image_prefix='', start_frame_idx=0):
    images_dir = os.path.join(images_dir, f'{i}')
    os.makedirs(images_dir, exist_ok=True)
    images = images.detach().cpu().numpy()
    for i, f in enumerate(images):
        imageio.imwrite(os.path.join(images_dir, image_prefix + f'{start_frame_idx + i:03d}.png'), to8b(f)) 
    if save_video:
        imageio.mimwrite(os.path.join(images_dir, os.path.basename(images_dir) + '.mp4'), to8b(images), fps=5, quality=8, macro_block_size = 8)
    
def read_images(img_dir, start_frame_idx=1, n_imgs=None):
    img_files = [os.path.join(img_dir, f) for f in natsort.natsorted(os.listdir(img_dir)) if f.endswith('png')]
    img_files = img_files[start_frame_idx-1:][:n_imgs]
    images = np.array([imageio.imread(f)/255. for f in img_files], dtype='float32')
    images = torch.tensor(images, device=device)
    return images

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
        
def load_meshes(meshes_dir, n_meshes=None):
    mesh_list = []
    obj_files = [os.path.join(meshes_dir, f) for f in natsort.natsorted(os.listdir(meshes_dir)) if f.endswith('obj')][:n_meshes]
    for f_obj in obj_files:
        verts, faces, _ = load_obj(f_obj, load_textures=False, device=device)
        mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
        mesh_list.append(mesh)
    if len(mesh_list) == 1:
        return mesh_list[0]
    return join_meshes_as_batch(mesh_list)

def load_template_mesh(template_filename):      
    verts, faces, aux = load_obj(template_filename, load_textures=False, device=device)
    return verts, faces.verts_idx, aux.verts_uvs

def load_pointclouds(pointclouds_dir, n_frames=None):
    pointcloud_files = [os.path.join(pointclouds_dir, f) for f in natsort.natsorted(os.listdir(pointclouds_dir)) if f.endswith('.obj')][:n_frames]
    pointcloud_points_list = []
    point_clouds_lengths = []
    for pointcloud_file in pointcloud_files:
        pointcloud_points, _, _ = load_obj(pointcloud_file, load_textures=False, device=device)
        point_clouds_lengths.append(pointcloud_points.shape[0])
        pointcloud_points_list.append(pointcloud_points)
    point_clouds = Pointclouds(points=pointcloud_points_list)
    point_clouds_lengths = torch.LongTensor(point_clouds_lengths).to(device)
    return point_clouds, point_clouds_lengths

def save_pointclouds(pointclouds, pointclouds_dir, i, temporal_sidelen):
    n_points = pointclouds.shape[1] // temporal_sidelen
    pointclouds_dir = os.path.join(pointclouds_dir, f'{i}')
    os.makedirs(pointclouds_dir, exist_ok=True)
    for surface_idx in range(temporal_sidelen):
        points = pointclouds[0,surface_idx * n_points:(surface_idx + 1) * n_points]
        save_obj(os.path.join(pointclouds_dir, f'{surface_idx:03d}.obj'), points, torch.tensor([]), verts_uvs=None, faces_uvs=None, texture_map=None)