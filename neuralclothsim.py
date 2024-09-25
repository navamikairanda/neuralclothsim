import os
from tqdm import trange
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, NamedTuple
from torch.utils.tensorboard import SummaryWriter  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_mgrid(sidelen: Tuple[int, int], stratified=False, dim=2):
    # Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    grid_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
    if stratified: 
        sidelen_x, sidelen_y = sidelen[0], sidelen[1]
    else: 
        sidelen_x, sidelen_y = sidelen[0] - 1, sidelen[1] - 1
    grid_coords[0, :, :, 0] = grid_coords[0, :, :, 0] / sidelen_x
    grid_coords[0, :, :, 1] = grid_coords[0, :, :, 1] / sidelen_y    
    grid_coords = torch.Tensor(grid_coords).to(device).view(-1, dim)
    return grid_coords

class CurvilinearSpace(NamedTuple):
    xi__1_max: float
    xi__2_max: float
    
class GridSampler(Dataset):
    def __init__(self, n_spatial_samples: int, curvilinear_space: CurvilinearSpace):
        self.n_spatial_samples = n_spatial_samples
                
        self.curvilinear_space = curvilinear_space
        self.spatial_sidelen = math.isqrt(n_spatial_samples)
        
        self.cell_curvilinear_coords = get_mgrid((self.spatial_sidelen, self.spatial_sidelen), stratified=True, dim=2)            

    def __len__(self):
        return 1
    
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
    def __init__(self, reference_geometry_name: str, boundary_condition_name: str, curvilinear_space: CurvilinearSpace):
        self.reference_geometry_name = reference_geometry_name
        self.boundary_condition_name = boundary_condition_name
        self.curvilinear_space = curvilinear_space
        self.boundary_support = 0.01

    def periodic_condition_and_normalization(self, curvilinear_coords: torch.Tensor) -> torch.Tensor:
        if self.reference_geometry_name in ['sleeve', 'skirt']:
            normalized_coords = torch.cat([(torch.cos(curvilinear_coords[...,0:1]) + 1)/2, (torch.sin(curvilinear_coords[...,0:1]) + 1)/2, curvilinear_coords[...,1:2]/self.curvilinear_space.xi__2_max], dim=2)
        else:
            normalized_coords = torch.cat([curvilinear_coords[...,0:1]/self.curvilinear_space.xi__1_max, curvilinear_coords[...,1:2]/self.curvilinear_space.xi__2_max], dim=2)
        return normalized_coords
    
    def dirichlet_condition(self, deformations: torch.Tensor, curvilinear_coords: torch.Tensor) -> torch.Tensor:        
        match self.boundary_condition_name:
            case 'top_left_fixed':
                top_left_corner = torch.exp(-(curvilinear_coords[...,0:1] ** 2 + (curvilinear_coords[...,1:2] - self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)
                deformations = deformations * (1 - top_left_corner)
            case 'top_bottom_rims_compression':                
                bottom_rim = torch.exp(-(curvilinear_coords[...,1:2] ** 2)/self.boundary_support)
                top_rim = torch.exp(-((curvilinear_coords[...,1:2] - self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)
                temporal_motion = 0.075 * torch.ones_like(curvilinear_coords[...,0:1])
                rim_displacement = torch.cat([torch.zeros_like(temporal_motion), temporal_motion, torch.zeros_like(temporal_motion)], dim=2)
                deformations = deformations * (1 - bottom_rim) * (1 - top_rim) - rim_displacement * top_rim + rim_displacement * bottom_rim
            case 'top_rim_fixed':
                top_rim = torch.exp(-((curvilinear_coords[...,1:2] - self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)
                deformations = deformations * (1 - top_rim)
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

class Mesh(NamedTuple):
    verts: torch.Tensor
    faces: torch.Tensor
    curvilinear_coords: torch.Tensor
    
class ReferenceMidSurface():
    def __init__(self, reference_geometry_name: str, test_n_spatial_samples: int, curvilinear_space: CurvilinearSpace):
        self.reference_geometry_name = reference_geometry_name
        test_spatial_sidelen = math.isqrt(test_n_spatial_samples)
        curvilinear_coords = get_mgrid((test_spatial_sidelen, test_spatial_sidelen), stratified=False, dim=2)[None]
        curvilinear_coords[...,0] *= curvilinear_space.xi__1_max
        curvilinear_coords[...,1] *= curvilinear_space.xi__2_max
        self.template_mesh = Mesh(verts=self(curvilinear_coords)[0], faces=generate_mesh_topology(test_spatial_sidelen), curvilinear_coords=curvilinear_coords)
        
    def __call__(self, curvilinear_coords: torch.Tensor) -> torch.Tensor:
        xi__1 = curvilinear_coords[...,0] 
        xi__2 = curvilinear_coords[...,1]
        match self.reference_geometry_name:
            case 'napkin':
                midsurface_positions = torch.stack([xi__1, xi__2, 0.* (xi__1**2 - xi__2**2)], dim=2)
            case 'sleeve':
                R = 0.25
                midsurface_positions = torch.stack([R * torch.cos(xi__1), xi__2, R * torch.sin(xi__1)], dim=2)
            case 'skirt':
                R_top, R_bottom, L = 0.2, 1.5, 1
                R = xi__2 * (R_top - R_bottom) / L + R_bottom
                midsurface_positions = torch.stack([R * torch.cos(xi__1), xi__2, R * torch.sin(xi__1)], dim=2)
        return midsurface_positions
    
from torch.nn.functional import normalize
import matplotlib.pyplot as plt

def get_plot_single_tensor(tensor):
    fig = plt.figure()
    ax = fig.gca()
    spatial_sidelen = math.isqrt(tensor.shape[1])
    pcolormesh = ax.pcolormesh(tensor.view(spatial_sidelen, spatial_sidelen).detach().cpu())
    fig.colorbar(pcolormesh, ax=ax)
    return fig

def get_plot_grid_tensor(tensor_1, tensor_2, tensor_3, tensor_4):
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0.22)
    (ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')
    spatial_sidelen = math.isqrt(tensor_1.shape[1])
    pcolormesh1 = ax1.pcolormesh(tensor_1.view(spatial_sidelen, spatial_sidelen).detach().cpu())
    pcolormesh2 = ax2.pcolormesh(tensor_2.view(spatial_sidelen, spatial_sidelen).detach().cpu())
    pcolormesh3 = ax3.pcolormesh(tensor_3.view(spatial_sidelen, spatial_sidelen).detach().cpu())
    pcolormesh4 = ax4.pcolormesh(tensor_4.view(spatial_sidelen, spatial_sidelen).detach().cpu())
    fig.colorbar(pcolormesh1, ax=ax1)
    fig.colorbar(pcolormesh2, ax=ax2)
    fig.colorbar(pcolormesh3, ax=ax3)
    fig.colorbar(pcolormesh4, ax=ax4)
    return fig

class ReferenceGeometry(): 
    def __init__(self, reference_midsurface: ReferenceMidSurface):
        self.reference_midsurface = reference_midsurface

    def __call__(self, curvilinear_coords: torch.Tensor, debug: bool):
        self.curvilinear_coords = curvilinear_coords
        self.midsurface_positions = self.reference_midsurface(self.curvilinear_coords)
        self.base_vectors()
        self.metric_tensor()
        self.curvature_tensor()
        self.christoffel_symbol()
        self.coord_transform()
        if debug:
            tb_writer.add_figure('metric_tensor', get_plot_grid_tensor(self.a_1_1, self.a_1_2, self.a_1_2, self.a_2_2))
            tb_writer.add_figure('curvature_tensor', get_plot_grid_tensor(self.b_1_1, self.b_1_2, self.b_2_1, self.b_2_2))
            
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

def compute_strain(deformations: torch.Tensor, ref_geometry: ReferenceGeometry, i: int, nonlinear_strain=True): 
    
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
    
    if not i % 200:
        tb_writer.add_figure(f'membrane_strain', get_plot_grid_tensor(epsilon_1_1, epsilon_1_2, epsilon_1_2, epsilon_2_2), i)
        tb_writer.add_figure(f'bending_strain', get_plot_grid_tensor(kappa_1_1, kappa_1_2, kappa_1_2, kappa_2_2), i)
    return Strain(epsilon_1_1, epsilon_1_2, epsilon_2_2, kappa_1_1, kappa_1_2, kappa_2_2)
    
class LinearMaterial():
    def __init__(self, mass_area_density, thickness, youngs_modulus, poissons_ratio, ref_geometry: ReferenceGeometry):
        self.mass_area_density = mass_area_density
        self.ref_geometry = ref_geometry
        self.poissons_ratio = poissons_ratio
        self.D = (youngs_modulus * thickness) / (1 - poissons_ratio ** 2)
        self.B = (thickness ** 2) * self.D / 12 
    
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

class Energy:
    def __init__(self, ref_geometry: ReferenceGeometry, material: LinearMaterial, train_n_spatial_samples: int, gravity_acceleration):
        self.ref_geometry = ref_geometry
        self.material = material
        self.external_load = torch.tensor(gravity_acceleration, device=device).expand(1, train_n_spatial_samples, 3) * material.mass_area_density
        
    def __call__(self, deformations: torch.Tensor, i: int) -> torch.Tensor:   
        strain = compute_strain(deformations, self.ref_geometry, i)
         
        hyperelastic_strain_energy_mid = self.material.compute_internal_energy(strain)
        external_energy_mid = torch.einsum('ijk,ijk->ij', self.external_load, deformations)
        mechanical_energy = (hyperelastic_strain_energy_mid - external_energy_mid) * torch.sqrt(self.ref_geometry.a)
        if not i % 200:
            tb_writer.add_histogram('param/hyperelastic_strain_energy', hyperelastic_strain_energy_mid, i) 
            tb_writer.add_figure(f'hyperelastic_strain_energy', get_plot_single_tensor(hyperelastic_strain_energy_mid), i)
        return mechanical_energy.mean()
                         
def test(ndf: Siren, reference_midsurface: ReferenceMidSurface, i: int):
    test_deformations = ndf(reference_midsurface.template_mesh.curvilinear_coords)
    test_deformed_positions = reference_midsurface.template_mesh.verts + test_deformations[0]
    tb_writer.add_mesh('simulated_states', test_deformed_positions[None], faces=reference_midsurface.template_mesh.faces[None], global_step=i)
     
def train(reference_geometry_name, boundary_condition_name, test_n_spatial_samples, n_iterations):
    log_dir = os.path.join('logs', reference_geometry_name)
            
    global tb_writer
    tb_writer = SummaryWriter(log_dir)

    curvilinear_space = CurvilinearSpace(xi__1_max=2 * math.pi if reference_geometry_name in ['sleeve', 'skirt'] else 1, xi__2_max=1)
    reference_midsurface = ReferenceMidSurface(reference_geometry_name, test_n_spatial_samples, curvilinear_space)
    boundary = Boundary(reference_geometry_name, boundary_condition_name, curvilinear_space)
    
    ndf = Siren(boundary, in_features=3 if reference_geometry_name in ['sleeve', 'skirt'] else 2).to(device)
    optimizer = torch.optim.Adam(lr=5e-6, params=ndf.parameters())
    
    reference_geometry = ReferenceGeometry(reference_midsurface)
    material = LinearMaterial(0.144, 0.0012, 5000, 0.25, reference_geometry)
    
    train_n_spatial_samples = 400
    energy = Energy(reference_geometry, material, train_n_spatial_samples, gravity_acceleration=[0, 0, 0] if reference_geometry_name == 'sleeve' else [0, 0, -9.8])    
    sampler = GridSampler(train_n_spatial_samples, curvilinear_space)
    dataloader = DataLoader(sampler, batch_size=1, num_workers=0)
        
    for i in trange(0, n_iterations):
        curvilinear_coords = next(iter(dataloader))
        reference_geometry(curvilinear_coords, i==0)
        deformations = ndf(reference_geometry.curvilinear_coords)
        loss = energy(deformations, i)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        tb_writer.add_scalar('loss/loss', loss, i)
        tb_writer.add_scalar('param/mean_deformation', deformations.mean(), i)
        if not i % 100:
            print(f'Iteration: {i}, loss: {loss}, mean_deformation: {deformations.mean()}')
            test(ndf, reference_midsurface, i)
    tb_writer.flush()

#train('napkin', 'top_left_fixed', 400, 601)
train('sleeve', 'top_bottom_rims_compression', 400, 601)
#train('skirt', 'top_rim_fixed', 400, 601)