import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange
from pytorch3d.io import save_obj, load_obj
from torch.utils.data import DataLoader

from config import device
from sampler import GridSampler
from diff_operators import jacobian
from modules import SirenReference, GELUReference
from sampler import sample_points_from_meshes
from pytorch3d.structures import Meshes
from reference_geometry import ReferenceGeometry
from helper import get_plot_single_tensor, get_plot_grid_tensor


def generate_mesh_topology(spatial_sidelen):
    rows = cols = spatial_sidelen
    last_face_index = cols * (rows - 1)
    
    first_face_bl = [0, cols, 1]  
    first_face_tr = [cols + 1, 1, cols]  
    all_faces = []
    for first_face in [first_face_bl, first_face_tr]:
        last_face = [i + last_face_index - 1 for i in first_face]
        faces = np.linspace(first_face, last_face, last_face_index)
        faces = np.reshape(faces, (rows - 1, cols, 3))
        faces = np.delete(faces, cols - 1, 1)
        faces = np.reshape(faces, (-1, 3))   
        all_faces.append(faces)
    return np.concatenate(all_faces, axis=0)

class ReferenceMidSurface():
    def __init__(self, args, tb_writer):
        self.reference_geometry_name = args.reference_geometry_name
        self.boundary_curvilinear_coords = None
        if args.reference_geometry_name in ['mesh']:
            self.vertices, faces, aux = load_obj(args.reference_geometry_source, load_textures=False, device=device)
            self.template_mesh = Meshes(verts=[self.vertices], faces=[faces.verts_idx]).cuda()
            self.curvilinear_coords = aux.verts_uvs
            self.faces = faces.verts_idx
            if args.boundary_condition_name == 'mesh_vertices':
                self.boundary_curvilinear_coords = self.curvilinear_coords[args.boundary_condition_vertices]
            self.fit_reference_mlp(args.reference_mlp_lrate, args.reference_mlp_n_iterations, args.test_spatial_sidelen, tb_writer)
            #self.vertices = self.midsurface(self.curvilinear_coords)[None] #verts# 
            self.temporal_coords = torch.linspace(0, args.end_time, args.test_temporal_sidelen, device=device)[:,None].repeat_interleave(self.vertices.shape[1], 0)[None]
        else:
            sampler = GridSampler(args.test_spatial_sidelen, args.test_temporal_sidelen, args.end_time, args.xi__1_scale, args.xi__2_scale, 'test')
            dataloader = DataLoader(sampler, batch_size=1, num_workers=0)
            self.curvilinear_coords, self.temporal_coords = next(iter(dataloader))
            self.vertices = self.midsurface(self.curvilinear_coords)
            self.faces = torch.tensor(generate_mesh_topology(args.test_spatial_sidelen), device=device)
            self.curvilinear_coords = self.curvilinear_coords[0]
        tb_writer.add_mesh('reference_state_fitted', self.vertices, faces=self.faces[None])
        save_obj(os.path.join(args.logging_dir, args.expt_name, 'reference_state_fitted.obj'), self.vertices[0], self.faces, verts_uvs=self.curvilinear_coords, faces_uvs=self.faces)
        #self.curvilinear_coords.requires_grad_(True)
        #ReferenceGeometry(self.curvilinear_coords[None], args.train_temporal_sidelen, 0, self, tb_writer, debug_ref_geometry=True)
        
    def fit_reference_mlp(self, reference_mlp_lrate, reference_mlp_n_iterations, spatial_sidelen, tb_writer):
        self.reference_mlp = SirenReference(first_omega_0=5., hidden_omega_0=5.).to(device)
        #self.reference_mlp = GELUReference(in_features=2, hidden_features=512, out_features=3, hidden_layers=3).to(device)
        reference_optimizer = torch.optim.Adam(lr=reference_mlp_lrate, params=self.reference_mlp.parameters())
        loss_fn = nn.L1Loss()   #nn.MSELoss()         
        for i in trange(reference_mlp_n_iterations):
            reference_optimizer.zero_grad()
            #with torch.no_grad(): 
            #    verts, uvs = sample_points_from_meshes(self.template_mesh, self.curvilinear_coords, 5000)
            fitted_verts = self.reference_mlp(self.curvilinear_coords[None])
            loss = loss_fn(fitted_verts, self.vertices)
            loss.backward()
            reference_optimizer.step()
            tb_writer.add_scalar('loss/reference_fitting_loss', loss.detach().item(), i)           
        
    def midsurface(self, curvilinear_coords):
        xi__1 = curvilinear_coords[...,0] 
        xi__2 = curvilinear_coords[...,1]
        if self.reference_geometry_name == 'rectangle':
            midsurface_positions = torch.stack([xi__1, xi__2, 0.* (xi__1**2 - xi__2**2)], dim=2)
            #midsurface_positions = torch.stack([xi__1 - 0.5, xi__2 - 0.5, 1.6 + 0.* (xi__1**2 - xi__2**2)], dim=2)            
            #midsurface_positions = torch.stack([xi__1 - 0.5, 1.0 + 0.* (xi__1**2 - xi__2**2), xi__2 - 0.5], dim=2)
            #midsurface_positions = torch.stack([xi__1, 1.0 + 0.* (xi__1**2 - xi__2**2), xi__2], dim=2) #collision sphere example
            #midsurface_positions = torch.stack([xi__1, 0.6001 + 0.* (xi__1**2 - xi__2**2), xi__2], dim=2) #collision bunny example        
        elif self.reference_geometry_name == 'cylinder':
            R = 0.25
            midsurface_positions = torch.stack([R * torch.cos(xi__1), xi__2, R * torch.sin(xi__1)], dim=2)
        elif self.reference_geometry_name == 'cone':
            R_top, R_bottom, L = 0.2, 1.5, 1
            R = xi__2 * (R_top - R_bottom) / L + R_bottom
            midsurface_positions = torch.stack([R * torch.cos(xi__1), xi__2, R * torch.sin(xi__1)], dim=2)
        elif self.reference_geometry_name == 'mesh':  
            midsurface_positions = self.reference_mlp(curvilinear_coords)
        return midsurface_positions