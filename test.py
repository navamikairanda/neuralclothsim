import torch 
from file_io import save_images, save_meshes, save_pointclouds
from config import device
import torch.nn as nn
from torch.nn.functional import relu
from modules import compute_sdf

def test(ndf, test_temporal_sidelen, meshes_dir, images_dir, i, tex_image_file, reference_midsurface, tb_writer):
    
    #test_deformations = ndf(reference_midsurface.curvilinear_coords.repeat(1, test_temporal_sidelen, 1), reference_midsurface.mesh_temporal_coords)
    test_deformations = ndf(reference_midsurface.curvilinear_coords.repeat(1, test_temporal_sidelen, 1), reference_midsurface.temporal_coords)
    test_deformed_positions = reference_midsurface.vertices.repeat(1, test_temporal_sidelen, 1) + test_deformations
    '''
    sdf, normal = compute_sdf(test_deformed_positions)
    eps = 0.0
    outward_displacement = torch.einsum('ij,ijk->ijk', relu(eps - sdf), normal)#relunn.GELU()
    test_deformed_positions = test_deformed_positions + outward_displacement
    '''
    #test_deformed_positions = reference_midsurface.midsurface(reference_midsurface.curvilinear_coords).repeat(1, test_temporal_sidelen, 1) + test_deformations
    tb_writer.add_mesh('simulated_states', test_deformed_positions.view(test_temporal_sidelen, -1, 3), faces=reference_midsurface.faces.repeat(test_temporal_sidelen, 1, 1), global_step=i)
    save_meshes(test_deformed_positions, reference_midsurface.faces, meshes_dir, i, test_temporal_sidelen, reference_midsurface.curvilinear_coords) 
