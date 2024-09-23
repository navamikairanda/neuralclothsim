import torch
import os
from tqdm import trange
from shutil import copyfile

from torch.utils.data import DataLoader
import utils.tb as tb
#torch.manual_seed(2) #Set seed for reproducible results

import torch.nn as nn
import numpy as np
import math
from torch.utils.data import Dataset

# Parts of the code (sample_points_from_meshes) borrowed from Meta Platforms, Inc.
from typing import Tuple, Union
from pytorch3d.structures import Meshes
from typing import NamedTuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_mgrid(sidelen: Union[Tuple[int], Tuple[int, int]], stratified=False, dim=2):
    # Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    if dim == 1:
        sidelen_x = sidelen[0] if stratified else sidelen[0] - 1
        grid_coords = np.stack(np.mgrid[:sidelen[0]], axis=-1)[None, ..., None].astype(np.float32)
        grid_coords[..., 0] = grid_coords[..., 0] / sidelen_x
    elif dim == 2:
        grid_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        if stratified: 
            sidelen_x, sidelen_y = sidelen[0], sidelen[1]
        else: 
            sidelen_x, sidelen_y = sidelen[0] - 1, sidelen[1] - 1
        grid_coords[0, :, :, 0] = grid_coords[0, :, :, 0] / sidelen_x
        grid_coords[0, :, :, 1] = grid_coords[0, :, :, 1] / sidelen_y
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)
    grid_coords = torch.Tensor(grid_coords).to(device).view(-1, dim)
    return grid_coords

class CurvilinearSpace(NamedTuple):
    xi__1_max: float
    xi__2_max: float
    
class Sampler(Dataset):
    def __init__(self, n_spatial_samples: int):
        self.n_spatial_samples = n_spatial_samples
    
    def __len__(self):
        return 1
                           
class GridSampler(Sampler):
    def __init__(self, n_spatial_samples: int, curvilinear_space: CurvilinearSpace):
        super().__init__(n_spatial_samples)
                
        self.curvilinear_space = curvilinear_space
        self.spatial_sidelen = math.isqrt(n_spatial_samples)
        
        self.cell_curvilinear_coords = get_mgrid((self.spatial_sidelen, self.spatial_sidelen), stratified=True, dim=2)            

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
         
        curvilinear_coords = self.cell_curvilinear_coords.clone()        
        
        t_rand_spatial = torch.rand([self.n_spatial_samples, 2], device=device) / self.spatial_sidelen        
        curvilinear_coords += t_rand_spatial
        
        curvilinear_coords[...,0] *= self.curvilinear_space.xi__1_max
        curvilinear_coords[...,1] *= self.curvilinear_space.xi__2_max
        curvilinear_coords.requires_grad_(True)
      
        return curvilinear_coords
    
class Boundary:
    def __init__(self, reference_geometry_name: str, boundary_condition_name: str, curvilinear_space: CurvilinearSpace, boundary_curvilinear_coords: torch.Tensor = None):
        self.reference_geometry_name = reference_geometry_name
        self.boundary_condition_name = boundary_condition_name
        self.curvilinear_space = curvilinear_space
        self.boundary_curvilinear_coords = boundary_curvilinear_coords
        self.boundary_support = 0.01

    def periodic_condition_and_normalization(self, curvilinear_coords: torch.Tensor) -> torch.Tensor:
        if self.reference_geometry_name in ['cylinder', 'cone']:
            normalized_coords = torch.cat([(torch.cos(curvilinear_coords[...,0:1]) + 1)/2, (torch.sin(curvilinear_coords[...,0:1]) + 1)/2, curvilinear_coords[...,1:2]/self.curvilinear_space.xi__2_max], dim=2)
        else:
            normalized_coords = torch.cat([curvilinear_coords[...,0:1]/self.curvilinear_space.xi__1_max, curvilinear_coords[...,1:2]/self.curvilinear_space.xi__2_max], dim=2)
        return normalized_coords
    
    def dirichlet_condition(self, deformations: torch.Tensor, curvilinear_coords: torch.Tensor) -> torch.Tensor:        
        match self.boundary_condition_name:
            case 'top_left_fixed':
                top_left_corner = torch.exp(-(curvilinear_coords[...,0:1] ** 2 + (curvilinear_coords[...,1:2] - self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)
                deformations = deformations * (1 - top_left_corner)
            case 'top_left_top_right_moved':
                top_left_corner = torch.exp(-(curvilinear_coords[...,0:1] ** 2 + (curvilinear_coords[...,1:2] - self.curvilinear_space.xi__1_max) ** 2)/self.boundary_support)
                top_right_corner = torch.exp(-((curvilinear_coords[...,0:1] - self.curvilinear_space.xi__1_max) ** 2 + (curvilinear_coords[...,1:2] - self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)                
                temporal_motion = 0.2 * torch.ones_like(curvilinear_coords[...,0:1])               
                corner_displacement = torch.cat([temporal_motion, torch.zeros_like(temporal_motion), torch.zeros_like(temporal_motion)], dim=2)                    
                deformations = deformations * (1 - top_left_corner) * (1 - top_right_corner) + corner_displacement * top_left_corner - corner_displacement * top_right_corner                                
            case 'adjacent_edges_fixed':
                left_edge = torch.exp(-(curvilinear_coords[...,0:1] ** 2)/self.boundary_support)
                right_edge = torch.exp(-(curvilinear_coords[...,1:2] ** 2)/self.boundary_support)
                deformations = deformations * (1 - left_edge) * (1 - right_edge)
            case 'nonboundary_handle_fixed':
                center_point = torch.exp(-((curvilinear_coords[...,0:1] - 0.5 * self.curvilinear_space.xi__1_max) ** 2 + (curvilinear_coords[...,1:2] - 0.7 * self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)
                deformations = deformations * (1 - center_point)
            case 'nonboundary_edge_fixed':
                center_edge = torch.exp(-((curvilinear_coords[...,0:1] - 0.5 * self.curvilinear_space.xi__1_max) ** 2)/self.boundary_support)
                deformations = deformations * (1 - center_edge)
            case 'top_bottom_rims_compression':                
                bottom_rim = torch.exp(-(curvilinear_coords[...,1:2] ** 2)/self.boundary_support)
                top_rim = torch.exp(-((curvilinear_coords[...,1:2] - self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)
                temporal_motion = 0.075 * torch.ones_like(curvilinear_coords[...,0:1])
                rim_displacement = torch.cat([torch.zeros_like(temporal_motion), temporal_motion, torch.zeros_like(temporal_motion)], dim=2)
                deformations = deformations * (1 - bottom_rim) * (1 - top_rim) - rim_displacement * top_rim + rim_displacement * bottom_rim
            case 'top_bottom_rims_torsion':
                self.boundary_support = 0.001
                bottom_rim = torch.exp(-(curvilinear_coords[...,1:2] ** 2)/self.boundary_support)
                top_rim = torch.exp(-((curvilinear_coords[...,1:2] - self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)
                R = 0.25
                rotation = math.pi / 4 #3 * math.pi / 4
                temporal_motion = torch.ones_like(curvilinear_coords[...,0:1]) * rotation
                top_rim_displacement = torch.cat([R * (torch.cos(curvilinear_coords[...,0:1] + temporal_motion) - torch.cos(curvilinear_coords[...,0:1])), torch.zeros_like(temporal_motion), R * (torch.sin(curvilinear_coords[...,0:1] + temporal_motion) - torch.sin(curvilinear_coords[...,0:1]))], dim=2)
                bottom_rim_displacement = torch.cat([R * (torch.cos(curvilinear_coords[...,0:1] - temporal_motion) - torch.cos(curvilinear_coords[...,0:1])), torch.zeros_like(temporal_motion), R * (torch.sin(curvilinear_coords[...,0:1] - temporal_motion) - torch.sin(curvilinear_coords[...,0:1])), ], dim=2)                
                deformations = deformations * (1 - bottom_rim) * (1 - top_rim) + top_rim_displacement * top_rim + bottom_rim_displacement * bottom_rim              
            case 'top_rim_fixed':
                top_rim = torch.exp(-((curvilinear_coords[...,1:2] - self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)
                deformations = deformations * (1 - top_rim)
            case 'top_rim_torsion':
                top_rim = torch.exp(-((curvilinear_coords[...,1:2] - self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)
                R_top = 0.2
                rotation = math.pi / 2
                temporal_motion = torch.ones_like(curvilinear_coords[...,0:1]) * rotation
                top_rim_displacement = torch.cat([R_top * (torch.cos(curvilinear_coords[...,0:1] + temporal_motion) - torch.cos(curvilinear_coords[...,0:1])), torch.zeros_like(temporal_motion), R_top * (torch.sin(curvilinear_coords[...,0:1] + temporal_motion) - torch.sin(curvilinear_coords[...,0:1]))], dim=2)
                deformations = deformations * (1 - top_rim) + top_rim_displacement * top_rim            
            case _:
                raise ValueError(f'Unknown boundary condition: {self.boundary_condition_name}')
        return deformations

class SineLayer(nn.Module):      
    def __init__(self, in_features: int, out_features: int, bias=True, is_first=False, omega_0=30.):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
   
class Siren(nn.Module):
    def __init__(self, boundary: Boundary, in_features=3, hidden_features=512, hidden_layers=5, out_features=3, outermost_linear=True, first_omega_0=30., hidden_omega_0=30.):
        super().__init__()
        self.boundary = boundary
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))    
        self.net = nn.Sequential(*self.net)
    
    def forward(self, curvilinear_coords: torch.Tensor) -> torch.Tensor:                

        normalized_coords = self.boundary.periodic_condition_and_normalization(curvilinear_coords)        
        deformations = self.net(normalized_coords)
        deformations = self.boundary.dirichlet_condition(deformations, curvilinear_coords)

        return deformations 

from torch.autograd import grad

def jacobian(y: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    ''' jacobian of y wrt x '''
    meta_batch_size, num_observations = y.shape[:2]
    jac = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[...,i].view(-1, 1)
        jac[:, :, i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True, allow_unused=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1
    return jac, status

from pytorch3d.io import save_obj, load_obj

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
    def __init__(self, args, curvilinear_space: CurvilinearSpace):
        self.reference_geometry_name = args.reference_geometry_name
        self.boundary_curvilinear_coords = None
        # for analytical surface, use equal number of samples along each curvilinear coordinate
        args.train_n_spatial_samples, args.test_n_spatial_samples = math.isqrt(args.train_n_spatial_samples) ** 2, math.isqrt(args.test_n_spatial_samples) ** 2
        test_spatial_sidelen = math.isqrt(args.test_n_spatial_samples)
        curvilinear_coords = get_mgrid((test_spatial_sidelen, test_spatial_sidelen), stratified=False, dim=2)[None]
        curvilinear_coords[...,0] *= curvilinear_space.xi__1_max
        curvilinear_coords[...,1] *= curvilinear_space.xi__2_max
        vertices = self(curvilinear_coords)[0]
        faces = torch.tensor(generate_mesh_topology(test_spatial_sidelen), device=device)
        texture = TexturesUV(maps=torch.empty(1, 1, 1, 1, device=device), faces_uvs=[faces], verts_uvs=curvilinear_coords)
        self.template_mesh = Meshes(verts=[vertices], faces=[faces], textures=texture).to(device)

        save_obj(os.path.join(args.logging_dir, args.expt_name, 'reference_state.obj'), self.template_mesh.verts_packed(), self.template_mesh.textures.faces_uvs_padded()[0], verts_uvs=self.template_mesh.textures.verts_uvs_padded()[0], faces_uvs=self.template_mesh.textures.faces_uvs_padded()[0])
        
    def __call__(self, curvilinear_coords: torch.Tensor) -> torch.Tensor:
        xi__1 = curvilinear_coords[...,0] 
        xi__2 = curvilinear_coords[...,1]
        match self.reference_geometry_name:
            case 'rectangle_xy': #vertical
                midsurface_positions = torch.stack([xi__1, xi__2, 0.* (xi__1**2 - xi__2**2)], dim=2)
            case 'rectangle_xz': #horizontal
                midsurface_positions = torch.stack([xi__1, 0.* (xi__1**2 - xi__2**2), xi__2], dim=2)
            case 'cylinder':
                R = 0.25
                midsurface_positions = torch.stack([R * torch.cos(xi__1), xi__2, R * torch.sin(xi__1)], dim=2)
            case 'cone':
                R_top, R_bottom, L = 0.2, 1.5, 1
                R = xi__2 * (R_top - R_bottom) / L + R_bottom
                midsurface_positions = torch.stack([R * torch.cos(xi__1), xi__2, R * torch.sin(xi__1)], dim=2)
            case _: 
                raise ValueError(f'Unknown reference_geometry_name {self.reference_geometry_name}')
        return midsurface_positions
    
from torch.nn.functional import normalize

from utils.plot import get_plot_grid_tensor

class ReferenceGeometry(): 
    def __init__(self, reference_midsurface: ReferenceMidSurface, n_spatial_samples: int):
        self.n_spatial_samples = n_spatial_samples 
        self.reference_midsurface = reference_midsurface

    def __call__(self, curvilinear_coords: torch.Tensor, debug: bool):
        self.curvilinear_coords = curvilinear_coords
        self.midsurface_positions = self.reference_midsurface(self.curvilinear_coords)
        self.base_vectors()
        self.metric_tensor()
        self.curvature_tensor()
        self.christoffel_symbol()
        self.coord_transform()
        if debug and tb.writer and self.reference_midsurface.reference_geometry_name != 'mesh':
            tb.writer.add_figure('metric_tensor', get_plot_grid_tensor(self.a_1_1[0], self.a_1_2[0],self.a_1_2[0], self.a_2_2[0]))
            tb.writer.add_figure('curvature_tensor', get_plot_grid_tensor(self.b_1_1[0,:self.n_spatial_samples], self.b_1_2[0,:self.n_spatial_samples],self.b_2_1[0,:self.n_spatial_samples], self.b_2_2[0,:self.n_spatial_samples]))
            
    def base_vectors(self):
        base_vectors = jacobian(self.midsurface_positions, self.curvilinear_coords)[0]
        self.a_1 = base_vectors[...,0]
        self.a_2 = base_vectors[...,1]
        self.a_3 = normalize(torch.linalg.cross(self.a_1, self.a_2), dim=2)
              
    def metric_tensor(self):
        self.a_1_1 = torch.einsum('ijk,ijk->ij', self.a_1, self.a_1)
        self.a_1_2 = torch.einsum('ijk,ijk->ij', self.a_1, self.a_2)
        self.a_2_2 = torch.einsum('ijk,ijk->ij', self.a_2, self.a_2)

        self.a = self.a_1_1 * self.a_2_2 - (self.a_1_2 ** 2)
        self.a__1__1 = self.a_2_2 / self.a
        self.a__2__2 = self.a_1_1 / self.a
        self.a__1__2 = -1 * self.a_1_2 / self.a
        self.a__2__1 = self.a__1__2
        self.a__1 = torch.einsum('ij,ijk->ijk', self.a__1__1, self.a_1) + torch.einsum('ij,ijk->ijk', self.a__1__2, self.a_2)
        self.a__2 = torch.einsum('ij,ijk->ijk', self.a__1__2, self.a_1) + torch.einsum('ij,ijk->ijk', self.a__2__2, self.a_2)        
    
    def curvature_tensor(self):        
        self.a_3pd = jacobian(self.a_3, self.curvilinear_coords)[0]
        self.b_1_1 = -1 * torch.einsum('ijk,ijk->ij', self.a_1, self.a_3pd[...,0])
        self.b_1_2 = -1 * torch.einsum('ijk,ijk->ij', self.a_1, self.a_3pd[...,1])
        self.b_2_2 = -1 * torch.einsum('ijk,ijk->ij', self.a_2, self.a_3pd[...,1])
        self.b_2_1 = self.b_1_2

        self.b_1__1 = self.b_1_1 * self.a__1__1 + self.b_1_2 * self.a__2__1
        self.b_1__2 = self.b_1_1 * self.a__1__2 + self.b_1_2 * self.a__2__2
        self.b_2__1 = self.b_2_1 * self.a__1__1 + self.b_2_2 * self.a__2__1
        self.b_2__2 = self.b_2_1 * self.a__1__2 + self.b_2_2 * self.a__2__2

    def coord_transform(self):
        with torch.no_grad():
            covariant_coord_2_cartesian = torch.stack([self.a__1, self.a__2, self.a_3], dim=3)
            self.cartesian_coord_2_covariant = torch.linalg.inv(covariant_coord_2_cartesian)        
    
    def christoffel_symbol(self):
        a_1pd = jacobian(self.a_1, self.curvilinear_coords)[0]
        a_2pd = jacobian(self.a_2, self.curvilinear_coords)[0]

        with torch.no_grad():
            self.gamma_11__1 = torch.einsum('ijk,ijk->ij', a_1pd[...,0], self.a__1)
            self.gamma_12__1 = torch.einsum('ijk,ijk->ij', a_1pd[...,1], self.a__1)
            self.gamma_22__1 = torch.einsum('ijk,ijk->ij', a_2pd[...,1], self.a__1)

            self.gamma_11__2 = torch.einsum('ijk,ijk->ij', a_1pd[...,0], self.a__2)
            self.gamma_12__2 = torch.einsum('ijk,ijk->ij', a_1pd[...,1], self.a__2)
            self.gamma_22__2 = torch.einsum('ijk,ijk->ij', a_2pd[...,1], self.a__2)
            self.gamma_21__1 = self.gamma_12__1
            self.gamma_21__2 = self.gamma_12__2

    @torch.no_grad()
    def elastic_tensor(self, poissons_ratio: float):
        H__1111 = poissons_ratio * self.a__1__1 * self.a__1__1 + 0.5 * (1 - poissons_ratio) * (self.a__1__1 * self.a__1__1 + self.a__1__1 * self.a__1__1)
        H__1112 = poissons_ratio * self.a__1__1 * self.a__1__2 + 0.5 * (1 - poissons_ratio) * (self.a__1__1 * self.a__1__2 + self.a__1__2 * self.a__1__1)
        H__1122 = poissons_ratio * self.a__1__1 * self.a__2__2 + 0.5 * (1 - poissons_ratio) * (self.a__1__2 * self.a__1__2 + self.a__1__2 * self.a__1__2)
        H__1212 = poissons_ratio * self.a__1__2 * self.a__1__2 + 0.5 * (1 - poissons_ratio) * (self.a__1__1 * self.a__2__2 + self.a__1__2 * self.a__2__1)
        H__1222 = poissons_ratio * self.a__1__2 * self.a__2__2 + 0.5 * (1 - poissons_ratio) * (self.a__1__2 * self.a__2__2 + self.a__1__2 * self.a__2__2)
        H__2222 = poissons_ratio * self.a__2__2 * self.a__2__2 + 0.5 * (1 - poissons_ratio) * (self.a__2__2 * self.a__2__2 + self.a__2__2 * self.a__2__2)
        return H__1111, H__1112, H__1122, H__1212, H__1222, H__2222
        
    def shell_base_vectors(self, xi__3: float):
        g_1_1 = self.a_1_1 - 2 * xi__3 * self.b_1_1
        g_1_2 = self.a_1_2 - 2 * xi__3 * self.b_1_2
        g_2_2 = self.a_2_2 - 2 * xi__3 * self.b_2_2
        
        g_1 = self.a_1 + xi__3 * self.a_3pd[...,0]
        g_2 = self.a_2 + xi__3 * self.a_3pd[...,1]
        g_covariant_matrix = torch.stack([torch.stack([g_1_1, g_1_2], dim=2), torch.stack([g_1_2, g_2_2], dim=2)], dim=2) 
        g_contravariant_matrix = torch.linalg.inv(g_covariant_matrix)
        g__1 = torch.einsum('ij,ijk->ijk', g_contravariant_matrix[...,0,0], g_1) + torch.einsum('ij,ijk->ijk', g_contravariant_matrix[...,0,1], g_2)
        g__2 = torch.einsum('ij,ijk->ijk', g_contravariant_matrix[...,1,0], g_1) + torch.einsum('ij,ijk->ijk', g_contravariant_matrix[...,1,1], g_2)

        g__1__1 = torch.einsum('ijk,ijl->ijkl', g__1, g__1)
        g__1__2 = torch.einsum('ijk,ijl->ijkl', g__1, g__2)
        g__2__1 = torch.einsum('ijk,ijl->ijkl', g__2, g__1)
        g__2__2 = torch.einsum('ijk,ijl->ijkl', g__2, g__2)
        return g__1__1, g__1__2, g__2__1, g__2__2    

class Strain(NamedTuple):
    epsilon_1_1: torch.Tensor
    epsilon_1_2: torch.Tensor
    epsilon_2_2: torch.Tensor
    kappa_1_1: torch.Tensor
    kappa_1_2: torch.Tensor
    kappa_2_2: torch.Tensor
    
def covariant_first_derivative_of_covariant_first_order_tensor(covariant_vectors, ref_geometry):
    vpd = jacobian(torch.stack(covariant_vectors, dim=2), ref_geometry.curvilinear_coords)[0]
    v_1, v_2 = covariant_vectors   
    
    v1cd1 = vpd[...,0,0] - v_1 * ref_geometry.gamma_11__1 - v_2 * ref_geometry.gamma_11__2
    v1cd2 = vpd[...,0,1] - v_1 * ref_geometry.gamma_12__1 - v_2 * ref_geometry.gamma_12__2
    v2cd1 = vpd[...,1,0] - v_1 * ref_geometry.gamma_21__1 - v_2 * ref_geometry.gamma_21__2
    v2cd2 = vpd[...,1,1] - v_1 * ref_geometry.gamma_22__1 - v_2 * ref_geometry.gamma_22__2
    
    return [v1cd1, v1cd2, v2cd1, v2cd2]

def covariant_first_derivative_of_covariant_second_order_tensor(covariant_matrix, ref_geometry):
    phipd = jacobian(torch.stack(covariant_matrix, dim=2), ref_geometry.curvilinear_coords)[0] 
    phi_1_1, phi_1_2, phi_2_1, phi_2_2 = covariant_matrix
    
    phi_1_1cd1 = phipd[...,0,0] - phi_1_1 * ref_geometry.gamma_11__1 - phi_2_1 * ref_geometry.gamma_11__2 - phi_1_1 * ref_geometry.gamma_11__1 - phi_1_2 * ref_geometry.gamma_11__2
    phi_1_1cd2 = phipd[...,0,1] - phi_1_1 * ref_geometry.gamma_12__1 - phi_2_1 * ref_geometry.gamma_12__2 - phi_1_1 * ref_geometry.gamma_12__1 - phi_1_2 * ref_geometry.gamma_12__2
    
    phi_1_2cd1 = phipd[...,1,0] - phi_1_2 * ref_geometry.gamma_11__1 - phi_2_2 * ref_geometry.gamma_11__2 - phi_1_1 * ref_geometry.gamma_21__1 - phi_1_2 * ref_geometry.gamma_21__2
    phi_1_2cd2 = phipd[...,1,1] - phi_1_2 * ref_geometry.gamma_12__1 - phi_2_2 * ref_geometry.gamma_12__2 - phi_1_1 * ref_geometry.gamma_22__1 - phi_1_2 * ref_geometry.gamma_22__2
    
    phi_2_1cd1 = phipd[...,2,0] - phi_1_1 * ref_geometry.gamma_21__1 - phi_2_1 * ref_geometry.gamma_21__2 - phi_2_1 * ref_geometry.gamma_11__1 - phi_2_2 * ref_geometry.gamma_11__2
    phi_2_1cd2 = phipd[...,2,1] - phi_1_1 * ref_geometry.gamma_22__1 - phi_2_1 * ref_geometry.gamma_22__2 - phi_2_1 * ref_geometry.gamma_12__1 - phi_2_2 * ref_geometry.gamma_12__2
    
    phi_2_2cd1 = phipd[...,3,0] - phi_1_2 * ref_geometry.gamma_21__1 - phi_2_2 * ref_geometry.gamma_21__2 - phi_2_1 * ref_geometry.gamma_21__1 - phi_2_2 * ref_geometry.gamma_21__2
    phi_2_2cd2 = phipd[...,3,1] - phi_1_2 * ref_geometry.gamma_22__1 - phi_2_2 * ref_geometry.gamma_22__2 - phi_2_1 * ref_geometry.gamma_22__1 - phi_2_2 * ref_geometry.gamma_22__2
    
    return phi_1_1cd1, phi_1_1cd2, phi_1_2cd1, phi_1_2cd2, phi_2_1cd1, phi_2_1cd2, phi_2_2cd1, phi_2_2cd2

def compute_strain(deformations: torch.Tensor, ref_geometry: ReferenceGeometry, i: int, i_debug: int, nonlinear_strain=True): 
    
    deformations_local = torch.einsum('ijkl,ijl->ijk', ref_geometry.cartesian_coord_2_covariant, deformations)
    u_1, u_2, u_3 = deformations_local[...,0], deformations_local[...,1], deformations_local[...,2]

    u1cd1, u1cd2, u2cd1, u2cd2 = covariant_first_derivative_of_covariant_first_order_tensor([u_1, u_2], ref_geometry) 
    
    phi_1_1 = u1cd1 - ref_geometry.b_1_1 * u_3
    phi_1_2 = u2cd1 - ref_geometry.b_2_1 * u_3
    phi_2_1 = u1cd2 - ref_geometry.b_1_2 * u_3
    phi_2_2 = u2cd2 - ref_geometry.b_2_2 * u_3

    u_3pd = jacobian(u_3[...,None], ref_geometry.curvilinear_coords)[0]
    phi_1_3 = u_3pd[...,0,0] + ref_geometry.b_1__1 * u_1 + ref_geometry.b_1__2 * u_2
    phi_2_3 = u_3pd[...,0,1] + ref_geometry.b_2__1 * u_1 + ref_geometry.b_2__2 * u_2
    
    epsilon_1_1_linear = phi_1_1
    epsilon_1_2_linear = 0.5 * (phi_1_2 + phi_2_1)
    epsilon_2_2_linear = phi_2_2
    
    phi_1_3cd1, phi_1_3cd2, phi_2_3cd1, phi_2_3cd2 = covariant_first_derivative_of_covariant_first_order_tensor([phi_1_3, phi_2_3], ref_geometry)
    
    kappa_1_1_linear = -1 * phi_1_3cd1 - ref_geometry.b_1__1 * phi_1_1 - ref_geometry.b_1__2 * phi_1_2 
    kappa_1_2_linear = -1 * phi_1_3cd2 - ref_geometry.b_2__1 * phi_1_1 - ref_geometry.b_2__2 * phi_1_2 
    kappa_2_2_linear = -1 * phi_2_3cd2 - ref_geometry.b_2__1 * phi_2_1 - ref_geometry.b_2__2 * phi_2_2 
    
    if nonlinear_strain:
        phi_1__1 = phi_1_1 * ref_geometry.a__1__1 + phi_1_2 * ref_geometry.a__2__1
        phi_1__2 = phi_1_1 * ref_geometry.a__1__2 + phi_1_2 * ref_geometry.a__2__2
        phi_2__1 = phi_2_1 * ref_geometry.a__1__1 + phi_2_2 * ref_geometry.a__2__1
        phi_2__2 = phi_2_1 * ref_geometry.a__1__2 + phi_2_2 * ref_geometry.a__2__2
    
        epsilon_1_1 = epsilon_1_1_linear + 0.5 * (phi_1_1 * phi_1__1 + phi_1_2 * phi_1__2 + phi_1_3 ** 2)
        epsilon_1_2 = epsilon_1_2_linear + 0.5 * (phi_1_1 * phi_2__1 + phi_1_2 * phi_2__2 + phi_1_3 * phi_2_3)
        epsilon_2_2 = epsilon_2_2_linear + 0.5 * (phi_2_1 * phi_2__1 + phi_2_2 * phi_2__2 + phi_2_3 ** 2)

        phi_3__1 = phi_1_3 * ref_geometry.a__1__1 + phi_2_3 * ref_geometry.a__2__1
        phi_3__2 = phi_1_3 * ref_geometry.a__1__2 + phi_2_3 * ref_geometry.a__2__2
        
        phi_1_1cd1, phi_1_1cd2, phi_1_2cd1, phi_1_2cd2, phi_2_1cd1, phi_2_1cd2, phi_2_2cd1, phi_2_2cd2 = covariant_first_derivative_of_covariant_second_order_tensor([phi_1_1, phi_1_2, phi_2_1, phi_2_2], ref_geometry)
    
        kappa_1_1 = kappa_1_1_linear + phi_3__1 * (phi_1_1cd1 + 0.5 * ref_geometry.b_1_1 * phi_1_3 - ref_geometry.b_1_1 * phi_1_3) + phi_3__2 * (phi_1_2cd1 + 0.5 * ref_geometry.b_1_1 * phi_2_3 - ref_geometry.b_1_2 * phi_1_3)
        kappa_1_2 = kappa_1_2_linear + phi_3__1 * (phi_1_1cd2 + 0.5 * ref_geometry.b_1_2 * phi_1_3 - ref_geometry.b_2_1 * phi_1_3) + phi_3__2 * (phi_1_2cd2 + 0.5 * ref_geometry.b_1_2 * phi_2_3 - ref_geometry.b_2_2 * phi_1_3)
        kappa_2_2 = kappa_2_2_linear + phi_3__1 * (phi_2_1cd1 + 0.5 * ref_geometry.b_2_2 * phi_1_3 - ref_geometry.b_2_1 * phi_2_3) + phi_3__2 * (phi_2_2cd2 + 0.5 * ref_geometry.b_2_2 * phi_2_3 - ref_geometry.b_2_2 * phi_2_3)
    else: 
        epsilon_1_1, epsilon_1_2, epsilon_2_2, kappa_1_1, kappa_1_2, kappa_2_2 = epsilon_1_1_linear, epsilon_1_2_linear, epsilon_2_2_linear, kappa_1_1_linear, kappa_1_2_linear, kappa_2_2_linear
    
    if not i % i_debug and tb.writer and ref_geometry.reference_midsurface.reference_geometry_name != 'mesh':
        tb.writer.add_figure(f'membrane_strain', get_plot_grid_tensor(epsilon_1_1[0,-ref_geometry.n_spatial_samples:], epsilon_1_2[0,-ref_geometry.n_spatial_samples:], epsilon_1_2[0,-ref_geometry.n_spatial_samples:], epsilon_2_2[0,-ref_geometry.n_spatial_samples:]), i)
        tb.writer.add_figure(f'bending_strain', get_plot_grid_tensor(kappa_1_1[0,-ref_geometry.n_spatial_samples:], kappa_1_2[0,-ref_geometry.n_spatial_samples:], kappa_1_2[0,-ref_geometry.n_spatial_samples:], kappa_2_2[0,-ref_geometry.n_spatial_samples:]), i)
    return Strain(epsilon_1_1, epsilon_1_2, epsilon_2_2, kappa_1_1, kappa_1_2, kappa_2_2)
    
class Material():
    def __init__(self, mass_area_density: float, thickness: float, ref_geometry: ReferenceGeometry):
        self.thickness = thickness
        self.mass_area_density = mass_area_density
        self.ref_geometry = ref_geometry
    
    def compute_internal_energy(self, strain: Strain) -> torch.Tensor:
        raise NotImplementedError
        
class LinearMaterial(Material):
    def __init__(self, args, ref_geometry: ReferenceGeometry):
        super().__init__(args.mass_area_density, args.thickness, ref_geometry)
        self.poissons_ratio = args.poissons_ratio
        self.D = (args.youngs_modulus * args.thickness) / (1 - args.poissons_ratio ** 2)
        self.B = (args.thickness ** 2) * self.D / 12 
    
    def compute_internal_energy(self, strain: Strain) -> torch.Tensor:
        H__1111, H__1112, H__1122, H__1212, H__1222, H__2222 = self.ref_geometry.elastic_tensor(self.poissons_ratio)
            
        n__1__1 = H__1111 * strain.epsilon_1_1 + 2 * H__1112 * strain.epsilon_1_2 + H__1122 * strain.epsilon_2_2
        n__1__2 = H__1112 * strain.epsilon_1_1 + 2 * H__1212 * strain.epsilon_1_2 + H__1222 * strain.epsilon_2_2
        n__2__1 = n__1__2
        n__2__2 = H__1122 * strain.epsilon_1_1 + 2 * H__1222 * strain.epsilon_1_2 + H__2222 * strain.epsilon_2_2

        m__1__1 = H__1111 * strain.kappa_1_1 + 2 * H__1112 * strain.kappa_1_2 + H__1122 * strain.kappa_2_2
        m__1__2 = H__1112 * strain.kappa_1_1 + 2 * H__1212 * strain.kappa_1_2 + H__1222 * strain.kappa_2_2
        m__2__1 = m__1__2
        m__2__2 = H__1122 * strain.kappa_1_1 + 2 * H__1222 * strain.kappa_1_2 + H__2222 * strain.kappa_2_2
        
        hyperelastic_strain_energy = 0.5 * (self.D * (strain.epsilon_1_1 * n__1__1 + strain.epsilon_1_2 * n__1__2 + strain.epsilon_1_2 * n__2__1 + strain.epsilon_2_2 * n__2__2) + self.B * (strain.kappa_1_1 * m__1__1 + strain.kappa_1_2 * m__1__2 + strain.kappa_1_2 * m__2__1 + strain.kappa_2_2 * m__2__2))
        
        return hyperelastic_strain_energy

class MaterialOrthotropy(NamedTuple):
    d_1: torch.Tensor
    d_2: torch.Tensor

class Energy:
    def __init__(self, ref_geometry: ReferenceGeometry, material: Material, gravity_acceleration: list, i_debug: int):
        self.ref_geometry = ref_geometry
        self.material = material
        external_load = torch.tensor(gravity_acceleration, device=device) * material.mass_area_density
        self.external_load = external_load.expand(1, ref_geometry.n_spatial_samples, 3)
        self.i_debug = i_debug
        
    def __call__(self, deformations: torch.Tensor, i: int) -> torch.Tensor:   
        strain = compute_strain(deformations, self.ref_geometry, i, self.i_debug)
         
        hyperelastic_strain_energy_mid = self.material.compute_internal_energy(strain)
        external_energy_mid = torch.einsum('ijk,ijk->ij', self.external_load, deformations)
        mechanical_energy = (hyperelastic_strain_energy_mid - external_energy_mid)
        mechanical_energy = mechanical_energy * torch.sqrt(self.ref_geometry.a)
            
        return mechanical_energy

import imageio

def save_meshes(positions, faces, meshes_dir, i, verts_uvs=None, tex_image_file=None):
    meshes_dir = os.path.join(meshes_dir, f'{i}')
    os.makedirs(meshes_dir, exist_ok=True)
    if tex_image_file is not None:
        texture_map = torch.tensor(imageio.imread(tex_image_file)/255., dtype=torch.float, device=device)
    else:
        texture_map = None
    save_obj(os.path.join(meshes_dir, 'simulated.obj'), positions[0], faces, verts_uvs=verts_uvs, faces_uvs=faces, texture_map=texture_map)

import sys
import logging

def get_logger(log_dir, expt_name):
    logger = logging.getLogger("NeuralClothSim")
    logger.setLevel(logging.DEBUG)

    stdoutHandler = logging.StreamHandler(stream=sys.stdout)    
    errHandler = logging.FileHandler(os.path.join(log_dir, f'{expt_name}.log'))

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s >>> %(message)s")
    stdoutHandler.setFormatter(fmt)
    errHandler.setFormatter(fmt)

    logger.addHandler(stdoutHandler)
    logger.addHandler(errHandler)
    return logger

import configargparse

def get_config_parser():
    parser = configargparse.ArgumentParser()
    parser.add('-c', '--config_filepath', required=True, is_config_file=True, help='config file path')
    parser.add_argument('-n', '--expt_name', type=str, required=True, help='experiment name; this will also be the name of subdirectory in logging_dir')

    # simulation parameters
    parser.add_argument('--reference_geometry_name', type=str, default='rectangle_xy', help='name of the reference geometry; can be rectangle_xy, rectangle_xz, cylinder, cone or mesh')
    parser.add_argument('--xi__1_max', type=float, default=2 * math.pi, help='max value for xi__1; 2 * pi for cylinder or cone, and Lx for rectangle. min value for xi__1 is assumed to be 0')
    parser.add_argument('--xi__2_max', type=float, default=1, help='max value for xi__2; Ly for all reference geometries. min value for xi__2 is assumed to be 0')
    parser.add_argument('--boundary_condition_name', type=str, default='top_left_fixed', help='name of the spatio-temporal boundary condition; can be one of top_left_fixed, top_left_top_right_moved, adjacent_edges_fixed, nonboundary_handle_fixed, nonboundary_edge_fixed for reference geometry as a rectangle, and top_bottom_rims_compression, top_bottom_rims_torsion for cylinder, top_rim_fixed, top_rim_torsion for cone, and mesh_vertices for mesh')
    parser.add_argument('--gravity_acceleration', type=float, nargs='+', default=[0,-9.8, 0], help='acceleration due to gravity')
    parser.add_argument('--trajectory', action='store_true', help='whether to simulate the trajectory from the reference state to the quasi-static state; otherwise, the quasistatic solutions are computed at all temporal samples')
    
    # additional parameters if the reference geometry is a mesh
    parser.add_argument('--reference_geometry_source', type=str, default='assets/textured_uniform_1_020.obj', help='source file for reference geometry')
    parser.add_argument('--reference_mlp_n_iterations', type=int, default=3000, help='number of iterations for fitting the reference geometry MLP')
    parser.add_argument('--reference_mlp_lrate', type=float, default=5e-6, help='learning rate for the reference geometry MLP')
    parser.add_argument('--reference_boundary_vertices', type=int, nargs='+', help='vertices for boundary condition on the reference geometry')
    
    # material parameters
    parser.add('-m', '--material_filepath', is_config_file=True, default='material/linear_1.ini', help='name of the material')
    parser.add_argument('--material_type', type=str, help='type of material; can be linear (isotropic) or nonlinear (orthotropic Clyde model)')
    parser.add_argument('--StVK', action='store_true', help='whether to use the St.Venant-Kirchhoff simplification of the Clyde material model')
    
    parser.add_argument('--mass_area_density', type=float, default=0.144, help='mass area density in kg/m^2 ')
    parser.add_argument('--thickness', type=float, default=0.0012, help='thickness in meters')
    
    # material parameters for a linear material
    parser.add_argument('--youngs_modulus', type=float, default=5000, help='Young\'s modulus')
    parser.add_argument('--poissons_ratio', type=float, default=0.25, help='Poisson\'s ratio')

    # material parameters for a nonlinear material [Clyde et al., 2017]
    parser.add_argument('--a11', type=float, help='a11')
    parser.add_argument('--a12', type=float, help='a12')
    parser.add_argument('--a22', type=float, help='a22')
    parser.add_argument('--G12', type=float, help='G12')
        
    parser.add_argument('--d', type=int, nargs='+', help='degree')
    parser.add_argument('--mu1', type=float, nargs='+', help='mu1')
    parser.add_argument('--mu2', type=float, nargs='+', help='mu2')
    parser.add_argument('--mu3', type=float, nargs='+', help='mu3')
    parser.add_argument('--mu4', type=float, nargs='+', help='mu4')
    parser.add_argument('--alpha1', type=float, nargs='+', help='alpha1')
    parser.add_argument('--alpha2', type=float, nargs='+', help='alpha2')
    parser.add_argument('--alpha3', type=float, nargs='+', help='alpha3')
    parser.add_argument('--alpha4', type=float, nargs='+', help='alpha4')
    
    parser.add_argument('--E_11_min', type=float, help='E_11_min')
    parser.add_argument('--E_11_max', type=float, help='E_11_max')
    parser.add_argument('--E_22_min', type=float, help='E_22_min')
    parser.add_argument('--E_22_max', type=float, help='E_22_max')
    parser.add_argument('--E_12_max', type=float, help='E_12_max')
    
    # training options
    parser.add_argument('--train_n_spatial_samples', type=int, default=400, help='N_omega, number of samples used for training; when the reference geometry is an analytical surface, the number of spatial grid samples along each curvilinear coordinate is square_root(N_omega)')
    parser.add_argument('--train_n_temporal_samples', type=int, default=10, help='N_t, number of temporal samples used for training')
    parser.add_argument('--lrate', type=float, default=5e-6, help='learning rate')
    parser.add_argument('--decay_lrate', action='store_true', default=True, help='whether to decay learning rate')
    parser.add_argument('--lrate_decay_steps', type=int, default=5000, help='learning rate decay steps')
    parser.add_argument('--lrate_decay_rate', type=float, default=0.1, help='learning rate decay rate')    
    parser.add_argument('--n_iterations', type=int, default=5001, help='total number of training iterations')
    
    # logging/saving options
    parser.add_argument('--logging_dir', type=str, default='logs', help='root directory for logging')
    parser.add_argument("--i_weights", type=int, default=400, help='frequency of saving NDF weights as checkpoints')
    parser.add_argument("--i_summary", type=int, default=100, help='frequency of logging losses')
    parser.add_argument("--i_test", type=int, default=100, help='frequency of evaluating NDF and saving simulated meshes during training')
    
    # debug options
    parser.add_argument('--debug', action='store_true', default=True, help='whether to run in debug mode; this will log the reference geometric quantities (e.g. metric and curvature tensor), the strains, and the simulated states to TensorBoard')
    parser.add_argument('--i_debug', type=int, default=200, help='frequency of Tensordboard logging')
    
    # reload options
    parser.add_argument('--no_reload', action='store_true', help='do not resume training from checkpoint, rather train from scratch')
    parser.add_argument("--i_ckpt", type=int, help='weight checkpoint to reload for resuming training or performing evaluation; if None, the latest checkpoint is used')
    parser.add_argument('--test_only', action='store_true', help='evaluate NDF from i_ckpt and save simulated meshes; do not resume training')
    
    # testing options
    parser.add_argument('--test_n_spatial_samples', type=int, default=400, help='N_omega, the number of samples used for evaluation when reference geometry is an analytical surface; if the reference geometry is instead a mesh, this argument is ignored, and the samples (vertices and faces) used for evaluation will match that of template mesh') 
    parser.add_argument('--test_n_temporal_samples', type=int, default=2, help='N_t, number of temporal samples used for evaluation')
                            
    return parser
                        
def test(ndf: Siren, reference_midsurface: ReferenceMidSurface, meshes_dir: str, i: int):
    
    test_deformations = ndf(reference_midsurface.template_mesh.textures.verts_uvs_padded())
    test_deformed_positions = reference_midsurface.template_mesh.verts_padded() + test_deformations
    if tb.writer:
        tb.writer.add_mesh('simulated_states', test_deformed_positions, faces=reference_midsurface.template_mesh.textures.faces_uvs_padded(), global_step=i)
    save_meshes(test_deformed_positions, reference_midsurface.template_mesh.textures.faces_uvs_padded()[0], meshes_dir, i, reference_midsurface.template_mesh.textures.verts_uvs_padded()[0]) 
        
def train():  
    args = get_config_parser().parse_args()
    log_dir = os.path.join(args.logging_dir, args.expt_name)
    meshes_dir = os.path.join(log_dir, 'meshes')   
    weights_dir = os.path.join(log_dir, 'weights')
        
    for dir in [log_dir, weights_dir]:
        os.makedirs(dir, exist_ok=True)
            
    logger = get_logger(log_dir, args.expt_name)
    logger.info(args)
    tb.set_tensorboard_writer(log_dir, args.debug)
    copyfile(args.config_filepath, os.path.join(log_dir, 'args.ini'))

    curvilinear_space = CurvilinearSpace(args.xi__1_max, args.xi__2_max)
    reference_midsurface = ReferenceMidSurface(args, curvilinear_space)
    boundary = Boundary(args.reference_geometry_name, args.boundary_condition_name, curvilinear_space, reference_midsurface.boundary_curvilinear_coords)
    
    ndf = Siren(boundary, in_features=3 if args.reference_geometry_name in ['cylinder', 'cone'] else 2).to(device)
    optimizer = torch.optim.Adam(lr=args.lrate, params=ndf.parameters())
    
    if args.i_ckpt is not None:
        ckpts = [os.path.join(weights_dir, f'{args.i_ckpt:06d}.tar')]
    else:
        ckpts = [os.path.join(weights_dir, f) for f in sorted(os.listdir(weights_dir)) if '0.tar' in f]
                
    logger.info(f'Found ckpts: {ckpts}')
    if len(ckpts) > 0 and not args.no_reload:
        logger.info(f'Resuming experiment {args.expt_name} from checkpoint {ckpts[-1]}')
        ckpt = torch.load(ckpts[-1])
        global_step = ckpt['global_step']
        ndf.load_state_dict(ckpt['siren_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    else: 
        logger.info(f'Starting experiment {args.expt_name}')
        global_step = 0
    
    if args.test_only:
        logger.info(f'Evaluating NDF from checkpoint {ckpts[-1]}')
        test(ndf, reference_midsurface, meshes_dir, global_step)
        return
    
    reference_geometry = ReferenceGeometry(reference_midsurface, args.train_n_spatial_samples)
    material = LinearMaterial(args, reference_geometry)
    energy = Energy(reference_geometry, material, args.gravity_acceleration, args.i_debug)
    
    sampler = GridSampler(args.train_n_spatial_samples, curvilinear_space)
    dataloader = DataLoader(sampler, batch_size=1, num_workers=0)
    
    if tb.writer:
        tb.writer.add_text('args', str(args))
        
    for i in trange(global_step, args.n_iterations):
        curvilinear_coords = next(iter(dataloader))
        reference_geometry(curvilinear_coords, i==0)
        deformations = ndf(reference_geometry.curvilinear_coords)                    
        mechanical_energy = energy(deformations, i)
        loss = mechanical_energy.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        if tb.writer:               
            tb.writer.add_scalar('loss/loss', loss, i)
            tb.writer.add_scalar('param/mean_deformation', deformations.mean(), i)
            
        if args.decay_lrate:
            new_lrate = args.lrate * args.lrate_decay_rate ** (i / args.lrate_decay_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate         
        
        if not i % args.i_summary:
            logger.info(f'Iteration: {i}, loss: {loss}, mean_deformation: {deformations.mean()}')      
            
        if not i % args.i_weights and i > 0:
            torch.save({
                'global_step': i,
                'siren_state_dict': ndf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(weights_dir, f'{i:06d}.tar'))
            
        if not i % args.i_test:            
            test(ndf, reference_midsurface, meshes_dir, i)
    tb.writer.flush()
    tb.writer.close()

if __name__=='__main__':
    train()