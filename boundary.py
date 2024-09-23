import math
import torch
from sampler import CurvilinearSpace

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
            case 'mesh_vertices':
                for i in range(self.boundary_curvilinear_coords.shape[0]):
                    boundary_point = torch.exp(-((curvilinear_coords[...,0:1] - self.boundary_curvilinear_coords[i][0]) ** 2 + (curvilinear_coords[...,1:2] - self.boundary_curvilinear_coords[i][1]) ** 2)/self.boundary_support)
                    deformations = deformations * (1 - boundary_point)
                deformations = deformations
            case _:
                raise ValueError(f'Unknown boundary condition: {self.boundary_condition_name}')
        return deformations