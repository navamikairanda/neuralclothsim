import os
import math
import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
from pytorch3d.io import save_obj, load_obj

from config_parser import device
from modules import SirenReference, GELUReference
from sampler import get_mgrid
from sampler import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import TexturesUV

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
            vertices, faces, aux = load_obj(args.reference_geometry_source, load_textures=False, device=device)
            texture = TexturesUV(maps=torch.empty(1, 1, 1, 1, device=device), faces_uvs=[faces.textures_idx], verts_uvs=[aux.verts_uvs])
            self.template_mesh = Meshes(verts=[vertices], faces=[faces.verts_idx], textures=texture).to(device)
            if args.boundary_condition_name == 'mesh_vertices':
                self.boundary_curvilinear_coords = self.template_mesh.textures.verts_uvs_padded()[0][args.reference_boundary_vertices]
            self.fit_reference_mlp(args.reference_mlp_lrate, args.reference_mlp_n_iterations, tb_writer)
            reference_mlp_verts_pred = self.midsurface(self.template_mesh.textures.verts_uvs_padded())
            self.template_mesh = self.template_mesh.update_padded(reference_mlp_verts_pred)
            #self.temporal_coords = torch.linspace(0, 1, args.test_n_temporal_samples, device=device)[:,None].repeat_interleave(self.template_mesh.num_verts_per_mesh().item(), 0)[None]
            self.temporal_coords = get_mgrid((args.test_n_temporal_samples,), stratified=False, dim=1).repeat_interleave(self.template_mesh.num_verts_per_mesh().item(), 0)[None]
        else:
            self.temporal_coords = get_mgrid((args.test_n_temporal_samples,), stratified=False, dim=1).repeat_interleave(args.test_n_spatial_samples, 0)[None]
            test_spatial_sidelen = math.isqrt(args.test_n_spatial_samples)
            curvilinear_coords = get_mgrid((test_spatial_sidelen, test_spatial_sidelen), stratified=False, dim=2)[None]
            curvilinear_coords[...,0] *= args.xi__1_max
            curvilinear_coords[...,1] *= args.xi__2_max
            vertices = self.midsurface(curvilinear_coords)[0]
            faces = torch.tensor(generate_mesh_topology(test_spatial_sidelen), device=device)
            texture = TexturesUV(maps=torch.empty(1, 1, 1, 1, device=device), faces_uvs=[faces], verts_uvs=curvilinear_coords)
            self.template_mesh = Meshes(verts=[vertices], faces=[faces], textures=texture).to(device)
        tb_writer.add_mesh('reference_state_fitted', self.template_mesh.verts_padded(), faces=self.template_mesh.textures.faces_uvs_padded())
        save_obj(os.path.join(args.logging_dir, args.expt_name, 'reference_state_fitted.obj'), self.template_mesh.verts_packed(), self.template_mesh.textures.faces_uvs_padded()[0], verts_uvs=self.template_mesh.textures.verts_uvs_padded()[0], faces_uvs=self.template_mesh.textures.faces_uvs_padded()[0])
        
    def fit_reference_mlp(self, reference_mlp_lrate, reference_mlp_n_iterations, tb_writer):
        #self.reference_mlp = SirenReference(first_omega_0=5., hidden_omega_0=5.).to(device)
        self.reference_mlp = GELUReference(in_features=2, hidden_features=512, out_features=3, hidden_layers=5).to(device)
        reference_optimizer = torch.optim.Adam(lr=reference_mlp_lrate, params=self.reference_mlp.parameters())
        loss_fn = nn.L1Loss()        
        for i in trange(reference_mlp_n_iterations):
            reference_optimizer.zero_grad()
            with torch.no_grad(): 
                verts, uvs = sample_points_from_meshes(self.template_mesh, 400)           
            loss = loss_fn(self.reference_mlp(uvs), verts)
            loss.backward()
            reference_optimizer.step()
            tb_writer.add_scalar('loss/reference_fitting_loss', loss.detach().item(), i)           
        
    def midsurface(self, curvilinear_coords: torch.Tensor) -> torch.Tensor:
        xi__1 = curvilinear_coords[...,0] 
        xi__2 = curvilinear_coords[...,1]
        if self.reference_geometry_name == 'rectangle_xy': #vertical
            midsurface_positions = torch.stack([xi__1, xi__2, 0.* (xi__1**2 - xi__2**2)], dim=2)
            #midsurface_positions = torch.stack([xi__1 - 0.5, xi__2 - 0.5, 1.6 + 0.* (xi__1**2 - xi__2**2)], dim=2)            
        elif self.reference_geometry_name == 'rectangle_xz': #horizontal
            midsurface_positions = torch.stack([xi__1, 0.* (xi__1**2 - xi__2**2), xi__2], dim=2) #non-boundary constraint
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