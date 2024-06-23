# Jupyter Notebook script

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import normalize
from torch.autograd import grad

import os
import sys
import pdb
import math
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import imageio

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

class SineLayer(nn.Module):      
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.):
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
    def __init__(self, time_scale, xi__1_scale, xi__2_scale, boundary_condition_name, reference_geometry_name, in_features=3, hidden_features=512, hidden_layers=5, out_features=3, outermost_linear=True, first_omega_0=30., hidden_omega_0=30.):
        super().__init__()
        self.time_scale = time_scale
        self.xi__1_scale = xi__1_scale
        self.xi__2_scale = xi__2_scale
        self.boundary_condition_name = boundary_condition_name
        self.reference_geometry_name = reference_geometry_name
        if self.reference_geometry_name in ['cylinder', 'cone']:
            in_features += 1
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
    
    def forward(self, curvilinear_coords, temporal_coords):
        
        ### Define initial and boundary condition ###
        initial_condition = temporal_coords ** 2
        boundary_support = 0.01
        if self.boundary_condition_name == 'top_left_fixed':
            top_left_corner = torch.exp(-(curvilinear_coords[...,0:1] ** 2 + (curvilinear_coords[...,1:2] - self.xi__2_scale) ** 2)/boundary_support)
        elif self.boundary_condition_name == 'two_rims_compression':
            bottom_rim = torch.exp(-(curvilinear_coords[...,1:2] ** 2)/boundary_support)
            top_rim = torch.exp(-((curvilinear_coords[...,1:2] - self.xi__2_scale) ** 2)/boundary_support)
            rim_displacement = torch.cat([torch.zeros_like(temporal_coords), torch.zeros_like(temporal_coords), 0.075 * temporal_coords], dim=2) 
        elif self.boundary_condition_name == 'top_rim_torsion':
            top_rim = torch.exp(-((curvilinear_coords[...,1:2] - self.xi__2_scale) ** 2)/boundary_support)
            R_top = 0.2
            top_rim_displacement = torch.cat([R_top * (torch.cos(curvilinear_coords[...,0:1] + temporal_coords * math.pi/2) - torch.cos(curvilinear_coords[...,0:1])), R_top * (torch.sin(curvilinear_coords[...,0:1] + temporal_coords * math.pi/2) - torch.sin(curvilinear_coords[...,0:1])), torch.zeros_like(temporal_coords)], dim=2)

        ### Normalize coordinates ###
        if self.reference_geometry_name in ['cylinder', 'cone']:
            normalized_coords = torch.cat([temporal_coords/self.time_scale, (torch.cos(curvilinear_coords[...,0:1]) +1)/2, (torch.sin(curvilinear_coords[...,0:1]) + 1)/2, curvilinear_coords[...,1:2]/self.xi__2_scale], dim=2)
        elif self.reference_geometry_name in ['rectangle']:
            normalized_coords = torch.cat([temporal_coords/self.time_scale, curvilinear_coords[...,0:1]/self.xi__1_scale, curvilinear_coords[...,1:2]/self.xi__2_scale], dim=2)
        output = self.net(normalized_coords)
        
        ### Apply boundary condition ###
        if self.boundary_condition_name == 'top_left_fixed':
            output = output * (1 - top_left_corner) * initial_condition
        elif self.boundary_condition_name == 'two_rims_compression':
            output = output * (1 - bottom_rim) * (1 - top_rim) * initial_condition - rim_displacement * top_rim + rim_displacement * bottom_rim
        elif self.boundary_condition_name == 'top_rim_torsion':
            output = output * (1 - top_rim) * initial_condition + top_rim_displacement * top_rim
        else: 
            output = output * initial_condition
        return output

def get_mgrid(sidelen, stratified=False, dim=2):
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
                       
class Sampler(Dataset):
    def __init__(self, spatial_sidelen, temporal_sidelen, time_scale, xi__1_scale, xi__2_scale, mode):
        super().__init__()
        self.mode = mode
        self.spatial_sidelen = spatial_sidelen
        self.temporal_sidelen = temporal_sidelen
                
        self.xi__1_scale = xi__1_scale
        self.xi__2_scale = xi__2_scale
        self.time_scale = time_scale
        if self.mode == 'train':
            self.cell_temporal_coords = get_mgrid((self.temporal_sidelen,), stratified=True, dim=1)
            self.cell_curvilinear_coords = get_mgrid((self.spatial_sidelen, self.spatial_sidelen), stratified=True, dim=2)
        elif self.mode == 'test':
            self.node_temporal_coords = get_mgrid((self.temporal_sidelen,), stratified=False, dim=1)
            self.node_curvilinear_coords = get_mgrid((self.spatial_sidelen, self.spatial_sidelen), stratified=False, dim=2)
            
    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
        if self.mode == 'test': 
            curvilinear_coords = self.node_curvilinear_coords.clone()
            temporal_coords = self.node_temporal_coords.clone() 
            temporal_coords *= self.time_scale
            curvilinear_coords[...,0] *= self.xi__1_scale 
            curvilinear_coords[...,1] *= self.xi__2_scale
        elif self.mode == 'train':            
            curvilinear_coords = self.cell_curvilinear_coords.clone() 
            temporal_coords = self.cell_temporal_coords.clone() 
            
            t_rand_temporal = torch.rand([self.temporal_sidelen, 1], device=device) / self.temporal_sidelen
            t_rand_spatial = torch.rand([self.spatial_sidelen**2, 2], device=device) / self.spatial_sidelen
            temporal_coords += t_rand_temporal
            curvilinear_coords += t_rand_spatial
            
            temporal_coords *= self.time_scale
            curvilinear_coords[...,0] *= self.xi__1_scale
            curvilinear_coords[...,1] *= self.xi__2_scale
            curvilinear_coords.requires_grad_(True)
            temporal_coords.requires_grad_(True)
            
        temporal_coords = temporal_coords.repeat_interleave(self.spatial_sidelen**2, 0)        
        return curvilinear_coords, temporal_coords            

def jacobian(y, x):
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

class ReferenceGeometry(): 
    def __init__(self, reference_geometry_name, curvilinear_coords, temporal_sidelen, mode, poissons_ratio=None):
        self.curvilinear_coords, self.temporal_sidelen = curvilinear_coords, temporal_sidelen        
        self.midsurface(reference_geometry_name)
        if mode == 'train': 
            self.base_vectors()
            self.metric_tensor()
            self.curvature_tensor()
            self.christoffel_symbol()
            self.elastic_tensor(poissons_ratio)
            self.coord_transform()   
        self.curvilinear_coords = self.curvilinear_coords.repeat(1, self.temporal_sidelen, 1)
        self.midsurface_positions = self.midsurface_positions.repeat(1, self.temporal_sidelen, 1) 
    
    def midsurface(self, reference_geometry_name):
        xi__1 = self.curvilinear_coords[...,0] 
        xi__2 = self.curvilinear_coords[...,1]
        if reference_geometry_name == 'rectangle':
            self.midsurface_positions = torch.stack([xi__1, xi__2, 0.* (xi__1**2 - xi__2**2)], dim=2)
        elif reference_geometry_name == 'cylinder':
            R = 0.25
            self.midsurface_positions = torch.stack([R * torch.cos(xi__1), R * torch.sin(xi__1), xi__2], dim=2)
        elif reference_geometry_name == 'cone':
            R_top, R_bottom, L = 0.2, 1.5, 1
            R = xi__2 * (R_top - R_bottom) / L + R_bottom
            self.midsurface_positions = torch.stack([R * torch.cos(xi__1), R * torch.sin(xi__1), xi__2], dim=2)
     
    def base_vectors(self):
        base_vectors = jacobian(self.midsurface_positions, self.curvilinear_coords)[0]
        self.a_1 = base_vectors[...,0]
        self.a_2 = base_vectors[...,1]
        self.a_3 = normalize(torch.cross(self.a_1, self.a_2), dim=2)
              
    def metric_tensor(self):    
        a_1_1 = torch.einsum('ijk,ijk->ij', self.a_1, self.a_1)
        a_1_2 = torch.einsum('ijk,ijk->ij', self.a_1, self.a_2)
        a_2_2 = torch.einsum('ijk,ijk->ij', self.a_2, self.a_2)

        self.a = a_1_1 * a_2_2 - (a_1_2 ** 2)
        self.a__1__1 = a_2_2 / self.a
        self.a__2__2 = a_1_1 / self.a
        self.a__1__2 = -1 * a_1_2 / self.a
        self.a__2__1 = self.a__1__2
        self.a__1 = torch.einsum('ij,ijk->ijk', self.a__1__1, self.a_1) + torch.einsum('ij,ijk->ijk', self.a__1__2, self.a_2)
        self.a__2 = torch.einsum('ij,ijk->ijk', self.a__1__2, self.a_1) + torch.einsum('ij,ijk->ijk', self.a__2__2, self.a_2)
        
        self.a__1__1 = self.a__1__1.repeat(1, self.temporal_sidelen)
        self.a__2__2 = self.a__2__2.repeat(1, self.temporal_sidelen)
        self.a__1__2 = self.a__1__2.repeat(1, self.temporal_sidelen)
        self.a__2__1 = self.a__2__1.repeat(1, self.temporal_sidelen)
        with torch.no_grad():
            self.a = self.a.repeat(1, self.temporal_sidelen)
    
    def curvature_tensor(self):        
        a_3pd = jacobian(self.a_3, self.curvilinear_coords)[0]
        self.b_1_1 = -1 * torch.einsum('ijk,ijk->ij', self.a_1, a_3pd[...,0])
        self.b_1_2 = -1 * torch.einsum('ijk,ijk->ij', self.a_1, a_3pd[...,1])
        self.b_2_2 = -1 * torch.einsum('ijk,ijk->ij', self.a_2, a_3pd[...,1])
        self.b_2_1 = self.b_1_2

        self.b_1_1 = self.b_1_1.repeat(1, self.temporal_sidelen)
        self.b_1_2 = self.b_1_2.repeat(1, self.temporal_sidelen)
        self.b_2_2 = self.b_2_2.repeat(1, self.temporal_sidelen)
        self.b_2_1 = self.b_2_1.repeat(1, self.temporal_sidelen)

        self.b_1__1 = self.b_1_1 * self.a__1__1 + self.b_1_2 * self.a__2__1
        self.b_1__2 = self.b_1_1 * self.a__1__2 + self.b_1_2 * self.a__2__2
        self.b_2__1 = self.b_2_1 * self.a__1__1 + self.b_2_2 * self.a__2__1
        self.b_2__2 = self.b_2_1 * self.a__1__2 + self.b_2_2 * self.a__2__2

    def coord_transform(self):
        with torch.no_grad():
            contravariant_coord_2_cartesian = torch.stack([self.a_1, self.a_2, self.a_3], dim=3)
            self.cartesian_coord_2_contravariant = torch.linalg.inv(contravariant_coord_2_cartesian)
            
            covariant_coord_2_cartesian = torch.stack([self.a__1, self.a__2, self.a_3], dim=3)
            self.cartesian_coord_2_covariant = torch.linalg.inv(covariant_coord_2_cartesian)
        
        self.cartesian_coord_2_contravariant = self.cartesian_coord_2_contravariant.repeat(1,self.temporal_sidelen,1,1)
        self.cartesian_coord_2_covariant = self.cartesian_coord_2_covariant.repeat(1,self.temporal_sidelen,1,1)

    @torch.no_grad()
    def elastic_tensor(self, poissons_ratio):
        H__1111 = poissons_ratio * self.a__1__1 * self.a__1__1 + 0.5 * (1 - poissons_ratio) * (self.a__1__1 * self.a__1__1 + self.a__1__1 * self.a__1__1)
        H__1112 = poissons_ratio * self.a__1__1 * self.a__1__2 + 0.5 * (1 - poissons_ratio) * (self.a__1__1 * self.a__1__2 + self.a__1__2 * self.a__1__1)
        H__1122 = poissons_ratio * self.a__1__1 * self.a__2__2 + 0.5 * (1 - poissons_ratio) * (self.a__1__2 * self.a__1__2 + self.a__1__2 * self.a__1__2)
        H__1212 = poissons_ratio * self.a__1__2 * self.a__1__2 + 0.5 * (1 - poissons_ratio) * (self.a__1__1 * self.a__2__2 + self.a__1__2 * self.a__2__1)
        H__1222 = poissons_ratio * self.a__1__2 * self.a__2__2 + 0.5 * (1 - poissons_ratio) * (self.a__1__2 * self.a__2__2 + self.a__1__2 * self.a__2__2)
        H__2222 = poissons_ratio * self.a__2__2 * self.a__2__2 + 0.5 * (1 - poissons_ratio) * (self.a__2__2 * self.a__2__2 + self.a__2__2 * self.a__2__2)
        self.H = [H__1111, H__1112, H__1122, H__1212, H__1222, H__2222]
    
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
        self.gamma_11__1 = self.gamma_11__1.repeat(1,self.temporal_sidelen)
        self.gamma_12__1 = self.gamma_12__1.repeat(1,self.temporal_sidelen)
        self.gamma_22__1 = self.gamma_22__1.repeat(1,self.temporal_sidelen)
        self.gamma_11__2 = self.gamma_11__2.repeat(1,self.temporal_sidelen)
        self.gamma_12__2 = self.gamma_12__2.repeat(1,self.temporal_sidelen)
        self.gamma_22__2 = self.gamma_22__2.repeat(1,self.temporal_sidelen)
        self.gamma_21__1 = self.gamma_21__1.repeat(1,self.temporal_sidelen)
        self.gamma_21__2 = self.gamma_21__2.repeat(1,self.temporal_sidelen)

class Material():
    def __init__(self, youngs_modulus=5000, poissons_ratio=0.25, mass_area_density=0.144, thickness=0.0012):
        self.poissons_ratio, self.mass_area_density = poissons_ratio, mass_area_density
        self.D = (youngs_modulus * thickness) / (1 - poissons_ratio ** 2)
        self.B = (thickness ** 2) * self.D / 12 

material = Material()
   
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

def compute_strain(deformations, ref_geometry, nonlinear=True): 
    
    u_1, u_2, u_3 = deformations[...,0], deformations[...,1], deformations[...,2]
    
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
    
    if nonlinear:
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
    
    epsilon_2_1 = epsilon_1_2
    kappa_2_1 = kappa_1_2
                
    return epsilon_1_1, epsilon_1_2, epsilon_2_1, epsilon_2_2, kappa_1_1, kappa_1_2, kappa_2_1, kappa_2_2

def compute_internal_energy(deformations, ref_geometry, material):
    epsilon_1_1, epsilon_1_2, epsilon_2_1, epsilon_2_2, kappa_1_1, kappa_1_2, kappa_2_1, kappa_2_2 = compute_strain(deformations, ref_geometry)
    
    H__1111, H__1112, H__1122, H__1212, H__1222, H__2222 = ref_geometry.H
    
    n__1__1 = H__1111 * epsilon_1_1 + H__1112 * epsilon_1_2 + H__1112 * epsilon_2_1 + H__1122 * epsilon_2_2
    n__1__2 = H__1112 * epsilon_1_1 + H__1212 * epsilon_1_2 + H__1212 * epsilon_2_1 + H__1222 * epsilon_2_2
    n__2__1 = n__1__2
    n__2__2 = H__1122 * epsilon_1_1 + H__1222 * epsilon_1_2 + H__1222 * epsilon_2_1 + H__2222 * epsilon_2_2

    m__1__1 = H__1111 * kappa_1_1 + H__1112 * kappa_1_2 + H__1112 * kappa_2_1 + H__1122 * kappa_2_2
    m__1__2 = H__1112 * kappa_1_1 + H__1212 * kappa_1_2 + H__1212 * kappa_2_1 + H__1222 * kappa_2_2
    m__2__1 = m__1__2
    m__2__2 = H__1122 * kappa_1_1 + H__1222 * kappa_1_2 + H__1222 * kappa_2_1 + H__2222 * kappa_2_2
    
    hyperelastic_strain_energy = 0.5 * (material.D * (epsilon_1_1 * n__1__1 + epsilon_1_2 * n__1__2 + epsilon_2_1 * n__2__1 + epsilon_2_2 * n__2__2) + material.B * (kappa_1_1 * m__1__1 + kappa_1_2 * m__1__2 + kappa_2_1 * m__2__1 + kappa_2_2 * m__2__2))
    return hyperelastic_strain_energy

def train(reference_geometry_name, boundary_condition_name, gravity_acceleration, end_time, n_iterations):
    if reference_geometry_name in ['rectangle']:
        xi__1_scale = xi__2_scale = 1
    elif reference_geometry_name in ['cylinder', 'cone']:
        xi__1_scale = 2 * math.pi
        xi__2_scale = 1
    time_scale = end_time

    spatial_sidelen, temporal_sidelen = 20, 20
    sampler = Sampler(spatial_sidelen, temporal_sidelen, time_scale, xi__1_scale, xi__2_scale, 'train')
    dataloader = DataLoader(sampler, batch_size=1, num_workers=0)

    
    ndf = Siren(time_scale, xi__1_scale, xi__2_scale, boundary_condition_name, reference_geometry_name).to(device)
    optimizer = torch.optim.Adam(lr=1e-5, params=ndf.parameters())
                
    losses = []
    external_load = torch.tensor(gravity_acceleration, device=device).expand(1, temporal_sidelen * spatial_sidelen**2, 3) * material.mass_area_density
    for i in trange(0, n_iterations):             
        curvilinear_coords, temporal_coords = next(iter(dataloader))
        ref_geometry = ReferenceGeometry(reference_geometry_name, curvilinear_coords, temporal_sidelen, 'train', material.poissons_ratio)
        deformations = ndf(ref_geometry.curvilinear_coords, temporal_coords)
        
        velocity = jacobian(deformations, temporal_coords)[0] 
        kinetic_energy = 0.5 * material.mass_area_density * torch.einsum('ijkl,ijkl->ij', velocity, velocity)
                
        external_energy = torch.einsum('ijk,ijk->ij', external_load, deformations)
        
        deformations = torch.einsum('ijkl,ijl->ijk', ref_geometry.cartesian_coord_2_covariant, deformations)
        internal_energy = compute_internal_energy(deformations, ref_geometry, material) 
        
        loss = ((kinetic_energy + internal_energy - external_energy) * torch.sqrt(ref_geometry.a)).mean()    
        losses.append(loss.detach().item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()
    ax.plot(losses)
    ax.set_xlabel("iteration", fontsize="16")
    ax.set_ylabel(' loss', fontsize="16")
    ax.set_title('Loss vs iterations', fontsize="16") 
    plt.show()
    
    return ndf

def evaluate(ndf, test_spatial_sidelen, test_temporal_sidelen):
    mode = 'test'    
    test_sampler = Sampler(test_spatial_sidelen, test_temporal_sidelen, ndf.time_scale, ndf.xi__1_scale, ndf.xi__2_scale, mode)
    test_dataloader = DataLoader(test_sampler, batch_size=1, num_workers=0)
    test_curvilinear_coords, test_temporal_coords = next(iter(test_dataloader))

    with torch.no_grad():
        test_ref_geometry = ReferenceGeometry(ndf.reference_geometry_name, test_curvilinear_coords, test_temporal_sidelen, mode)
        test_deformations = ndf(test_ref_geometry.curvilinear_coords, test_temporal_coords) 
        test_deformed_positions = test_ref_geometry.midsurface_positions + test_deformations
    return test_deformed_positions
  
from pytorch3d.renderer import (
    PerspectiveCameras, 
    MeshRenderer, 
    MeshRasterizer, 
    RasterizationSettings,
    HardPhongShader, 
    TexturesVertex,
    TexturesUV,
    look_at_view_transform,
    PointLights
)
from pytorch3d.structures import Meshes

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

def render_meshes(vertices, temporal_sidelen, spatial_sidelen, reference_geometry_name):   
    if reference_geometry_name == 'rectangle':        
        lights = PointLights(device=device, location=[[0.0, 0.0, 0.6]])
        #object_pos = torch.tensor([[0.5, 0, 0.]], device=device)
        object_pos = torch.tensor([[0.5, 0.5, 0.]], device=device)
        x_dir = ((0, 1, 0),)
        #camera_pos = torch.tensor([[0, 0, 1.2]], device=device)
        camera_pos = torch.tensor([[0, 0, 1.2]], device=device)
        verts_features = torch.tensor([1.0, 0.6, 0.6], device=device)
    elif reference_geometry_name == 'cylinder':        
        lights = PointLights(device=device, location=[[0, -1., 0.5]])
        object_pos = torch.tensor([[0., 0., 0.5]], device=device)
        x_dir = ((0, 0, 1),)        
        camera_pos = torch.tensor([[0, -1., 0.5]], device=device)
        verts_features = torch.tensor([0.6, 0.8, 1.0], device=device)
    elif reference_geometry_name == 'cone':        
        lights = PointLights(device=device, location=[[0, -2.25, 0.5]])
        object_pos = torch.tensor([[0., 0., 0.2]], device=device)
        x_dir = ((0, 0, 1),)        
        camera_pos = torch.tensor([[0, -2.25, 0.5]], device=device)
        verts_features = torch.tensor([0.56, 0.59, 0.48], device=device)
    
    vertices = vertices.reshape((temporal_sidelen, -1, 3))
    faces = torch.tensor(generate_mesh_topology(spatial_sidelen), device=device).repeat(temporal_sidelen, 1, 1)
    textures = TexturesVertex(verts_features=verts_features.expand(temporal_sidelen, vertices.shape[1],-1))
    meshes = Meshes(vertices, faces, textures=textures)
    R, T = look_at_view_transform(eye=camera_pos, up=x_dir, at=object_pos) 
    cameras = PerspectiveCameras(T=T, R=R, device=device)
    raster_settings = RasterizationSettings(image_size=1024)    
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = HardPhongShader(device=device, cameras=cameras, lights=lights)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    images = renderer(meshes)
    return images.detach().cpu().numpy()

def visualise(ndf, test_temporal_sidelen, test_spatial_sidelen):
    test_deformed_positions = evaluate(ndf, test_spatial_sidelen, test_temporal_sidelen)
    images = render_meshes(test_deformed_positions, test_temporal_sidelen, test_spatial_sidelen, ndf.reference_geometry_name)
    f = '{}.mp4'.format(ndf.reference_geometry_name)
    to8b = lambda x : (255 * np.clip(x, 0, 1)).astype(np.uint8)
    imageio.mimwrite(f, to8b(images), fps=5)     #Tell how slow it is rendered
    ''' 
    mp4 = open(f,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    HTML("""
    <video width=400 controls autoplay loop>
        <source src="%s" type="video/mp4">
    </video>
    """ % data_url)
    ''' 
    
'''
from IPython.display import HTML
from base64 import b64encode
'''

ndf = train('rectangle', 'top_left_fixed', [0,-9.8, 0], 1, 50)
visualise(ndf, 20, 30)

'''
ndf = train('cylinder', 'two_rims_compression', [0., 0., 0.], 1, 50)
visualise(ndf, 20, 50)

ndf = train('cone', 'top_rim_torsion', [0., 0, -9.8], 1, 1000)
visualise(ndf, 20, 500)
'''