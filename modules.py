import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.functional import normalize
from config_parser import device

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
    def __init__(self, xi__1_scale, xi__2_scale, boundary_condition_name, reference_geometry_name, in_features=3, hidden_features=512, hidden_layers=5, out_features=3, outermost_linear=True, first_omega_0=30., hidden_omega_0=30., k=10., boundary_curvilinear_coords=None):
        super().__init__()
        self.k = k
        self.xi__1_scale = xi__1_scale
        self.xi__2_scale = xi__2_scale
        self.boundary_condition_name = boundary_condition_name
        self.boundary_curvilinear_coords = boundary_curvilinear_coords
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
                #final_linear.weight.data = final_linear.weight.data / 1.e2
                #final_linear.bias.data = final_linear.bias.data / 1.e2
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))    
        self.net = nn.Sequential(*self.net)
    
    def forward(self, curvilinear_coords, temporal_coords):
        
        ### Define initial and boundary condition ###
        initial_condition = temporal_coords ** 2
        #initial_condition = torch.tanh(self.k * (temporal_coords ** 2))
        boundary_support = 0.01
        if self.boundary_condition_name == 'top_left_fixed':
            top_left_corner = torch.exp(-(curvilinear_coords[...,0:1] ** 2 + (curvilinear_coords[...,1:2] - self.xi__2_scale) ** 2)/boundary_support)
        elif self.boundary_condition_name == 'two_rims_compression':
            bottom_rim = torch.exp(-(curvilinear_coords[...,1:2] ** 2)/boundary_support)
            top_rim = torch.exp(-((curvilinear_coords[...,1:2] - self.xi__2_scale) ** 2)/boundary_support)
            #rim_displacement = torch.cat([torch.zeros_like(temporal_coords), 0.075 * temporal_coords, torch.zeros_like(temporal_coords)], dim=2)
            rim_displacement = torch.cat([torch.zeros_like(temporal_coords), 0.075 * torch.ones_like(temporal_coords), torch.zeros_like(temporal_coords)], dim=2)
        elif self.boundary_condition_name == 'top_rim_fixed':
            top_rim = torch.exp(-((curvilinear_coords[...,1:2] - self.xi__2_scale) ** 2)/boundary_support)         
        elif self.boundary_condition_name == 'top_rim_torsion':
            top_rim = torch.exp(-((curvilinear_coords[...,1:2] - self.xi__2_scale) ** 2)/boundary_support)
            R_top = 0.2
            top_rim_displacement = torch.cat([R_top * (torch.cos(curvilinear_coords[...,0:1] + temporal_coords * math.pi/2) - torch.cos(curvilinear_coords[...,0:1])), torch.zeros_like(temporal_coords), R_top * (torch.sin(curvilinear_coords[...,0:1] + temporal_coords * math.pi/2) - torch.sin(curvilinear_coords[...,0:1]))], dim=2)
        elif self.boundary_condition_name == 'center':
            center_point = torch.exp(-((curvilinear_coords[...,0:1] - 0.5 * self.xi__1_scale) ** 2 + (curvilinear_coords[...,1:2] - 0.7 * self.xi__2_scale) ** 2)/boundary_support)
        elif self.boundary_condition_name == 'top_left_top_right_drape':
            top_left_corner = torch.exp(-(curvilinear_coords[...,0:1] ** 2 + (curvilinear_coords[...,1:2] - self.xi__1_scale) ** 2)/0.01)
            top_right_corner = torch.exp(-((curvilinear_coords[...,0:1] - self.xi__1_scale) ** 2 + (curvilinear_coords[...,1:2] - self.xi__2_scale) ** 2)/0.01)
            #top_left_corner = torch.exp(-(curvilinear_coords[...,0:1] ** 2 + curvilinear_coords[...,1:2] ** 2)/0.01)
            #top_right_corner = torch.exp(-((curvilinear_coords[...,0:1] - self.xi__1_scale) ** 2 + curvilinear_coords[...,1:2] ** 2)/0.01)
            #corner_displacement = torch.cat([0.2 * temporal_coords, torch.zeros_like(temporal_coords), torch.zeros_like(temporal_coords)], dim=2)
            corner_displacement = torch.cat([0.2 * torch.ones_like(temporal_coords), torch.zeros_like(temporal_coords), torch.zeros_like(temporal_coords)], dim=2)

        ### Normalize coordinates ###
        if self.reference_geometry_name in ['cylinder', 'cone']:
            normalized_coords = torch.cat([temporal_coords, (torch.cos(curvilinear_coords[...,0:1]) + 1)/2, (torch.sin(curvilinear_coords[...,0:1]) + 1)/2, curvilinear_coords[...,1:2]/self.xi__2_scale], dim=2)
        elif self.reference_geometry_name in ['rectangle', 'mesh']:
            normalized_coords = torch.cat([temporal_coords, curvilinear_coords[...,0:1]/self.xi__1_scale, curvilinear_coords[...,1:2]/self.xi__2_scale], dim=2)
        output = self.net(normalized_coords)
        
        ### Apply boundary condition ###
        if self.boundary_condition_name == 'top_left_fixed':
            output = output * (1 - top_left_corner) #* initial_condition
        elif self.boundary_condition_name == 'two_rims_compression':
            #output = output * (1 - bottom_rim) * (1 - top_rim) * initial_condition - rim_displacement * top_rim + rim_displacement * bottom_rim
            output = output * (1 - bottom_rim) * (1 - top_rim) - rim_displacement * top_rim + rim_displacement * bottom_rim
        elif self.boundary_condition_name == 'top_rim_fixed':
            output = output * (1 - top_rim) #* initial_condition
        elif self.boundary_condition_name == 'top_rim_torsion':
            output = output * (1 - top_rim) * initial_condition + top_rim_displacement * top_rim
        elif self.boundary_condition_name == 'center':
            output = output * (1 - center_point) * initial_condition
        elif self.boundary_condition_name == 'top_left_top_right_drape':
            #output = output * (1 - top_left_corner) * (1 - top_right_corner) * initial_condition + corner_displacement * top_left_corner - corner_displacement * top_right_corner
            output = output * (1 - top_left_corner) * (1 - top_right_corner) + corner_displacement * top_left_corner - corner_displacement * top_right_corner
        elif self.boundary_condition_name == 'mesh_vertices':
            for i in range(self.boundary_curvilinear_coords.shape[0]):
                output = output * (1 - torch.exp(-((curvilinear_coords[...,0:1] - self.boundary_curvilinear_coords[i][0]) ** 2 + (curvilinear_coords[...,1:2] - self.boundary_curvilinear_coords[i][1]) ** 2)/boundary_support))
        else: 
            pass #* initial_condition
        return output
    
class SirenReference(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=3, out_features=3, outermost_linear=True, first_omega_0=30., hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, np.sqrt(6 / hidden_features) / hidden_omega_0)
                #final_linear.weight.data = final_linear.weight.data / 1.e2
                #final_linear.bias.data = final_linear.bias.data / 1.e2
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))    
        self.net = nn.Sequential(*self.net)
    
    def forward(self, curvilinear_coords):
        output = self.net(curvilinear_coords)
        return output

class GELUReference(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
    
        super().__init__()
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))

        for i in range(hidden_layers-1):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.GELU())
            #self.net.append(nn.SiLU())
            #self.net.append(nn.ReLU())
            #self.net.append(nn.Tanh())
        
        self.net.append(nn.Linear(hidden_features, hidden_features))
        self.net.append(nn.GELU())
        #self.net.append(nn.SiLU())

        final_linear = nn.Linear(hidden_features, out_features)
            
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

    def forward(self, curvilinear_coords):
        output = self.net(curvilinear_coords)
        return output

def compute_sdf(positions):
    center = torch.tensor([0.5, 0.5, 0.5], device=device)
    radius = 0.3 #0.4, 0.1
    #sdf = ((positions - center) ** 2).sum(2) - radius ** 2
    sdf = torch.sqrt(((positions - center) ** 2).sum(2)) - radius
    normal = normalize(positions - center, dim=2) #jacobian(sdf, positions)[0][...,0,:]
    #torch.einsum('ij,ijk->ijk', relu(eps - sdf), normal)
    return sdf, normal 