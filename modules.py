import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import normalize
from config_parser import device
from boundary import Boundary

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
    
    def forward(self, curvilinear_coords: torch.Tensor, temporal_coords: torch.Tensor):                

        normalized_coords = self.boundary.periodic_condition_and_normalization(curvilinear_coords, temporal_coords)        
        deformations = self.net(normalized_coords)
        deformations = self.boundary.dirichlet_condition(deformations, curvilinear_coords, temporal_coords)

        return deformations    

class GELUReference(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
    
        super().__init__()
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))

        for i in range(hidden_layers-1):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.GELU())
        
        self.net.append(nn.Linear(hidden_features, hidden_features))
        self.net.append(nn.GELU())

        final_linear = nn.Linear(hidden_features, out_features)
            
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

    def forward(self, curvilinear_coords):
        output = self.net(curvilinear_coords)
        return output