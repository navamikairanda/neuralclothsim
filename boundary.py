import math
import torch

class Boundary:
    def __init__(self, reference_geometry_name, boundary_condition_name, curvilinear_space, boundary_curvilinear_coords=None):
        self.reference_geometry_name = reference_geometry_name
        self.boundary_condition_name = boundary_condition_name
        self.curvilinear_space = curvilinear_space
        self.boundary_curvilinear_coords = boundary_curvilinear_coords
        self.boundary_support = 0.01

    def periodic_condition_and_normalization(self, curvilinear_coords, temporal_coords):
        if self.reference_geometry_name in ['cylinder', 'cone']:
            normalized_coords = torch.cat([temporal_coords, (torch.cos(curvilinear_coords[...,0:1]) + 1)/2, (torch.sin(curvilinear_coords[...,0:1]) + 1)/2, curvilinear_coords[...,1:2]/self.curvilinear_space.xi__2_max], dim=2)
        else:
            normalized_coords = torch.cat([temporal_coords, curvilinear_coords[...,0:1]/self.curvilinear_space.xi__1_max, curvilinear_coords[...,1:2]/self.curvilinear_space.xi__2_max], dim=2)
        return normalized_coords
    
    def dirichlet_condition(self, deformations, curvilinear_coords, temporal_coords):
        initial_condition = temporal_coords ** 2 # or torch.ones_like(temporal_coords)
        match self.boundary_condition_name:
            case 'top_left_fixed':
                top_left_corner = torch.exp(-(curvilinear_coords[...,0:1] ** 2 + (curvilinear_coords[...,1:2] - self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)
                deformations = deformations * (1 - top_left_corner) * initial_condition
            case 'top_left_top_right_moved':
                top_left_corner = torch.exp(-(curvilinear_coords[...,0:1] ** 2 + (curvilinear_coords[...,1:2] - self.curvilinear_space.xi__1_max) ** 2)/self.boundary_support)
                top_right_corner = torch.exp(-((curvilinear_coords[...,0:1] - self.curvilinear_space.xi__1_max) ** 2 + (curvilinear_coords[...,1:2] - self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)                
                corner_displacement = torch.cat([0.2 * torch.ones_like(temporal_coords), torch.zeros_like(temporal_coords), torch.zeros_like(temporal_coords)], dim=2)                
                deformations = deformations * (1 - top_left_corner) * (1 - top_right_corner) + corner_displacement * top_left_corner - corner_displacement * top_right_corner
                #corner_displacement = torch.cat([0.2 * temporal_coords, torch.zeros_like(temporal_coords), torch.zeros_like(temporal_coords)], dim=2)
                #deformations = deformations * (1 - top_left_corner) * (1 - top_right_corner) * initial_condition + corner_displacement * top_left_corner - corner_displacement * top_right_corner
            case 'adjacent_edges_fixed':
                #boundary_condition = curvilinear_coords[...,0][...,None] * curvilinear_coords[...,1][...,None]
                left_edge = torch.exp(-(curvilinear_coords[...,0:1] ** 2)/self.boundary_support)
                right_edge = torch.exp(-((curvilinear_coords[...,0:1] - self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)
                #boundary_condition = curvilinear_coords[...,0:1] * curvilinear_coords[...,1]
                #output = output * boundary_condition * initial_condition
                deformations = deformations * (1 - left_edge) * (1 - right_edge) * initial_condition
            case 'nonboundary_handle_fixed':
                center_point = torch.exp(-((curvilinear_coords[...,0:1] - 0.5 * self.curvilinear_space.xi__1_max) ** 2 + (curvilinear_coords[...,1:2] - 0.7 * self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)
                deformations = deformations * (1 - center_point) * initial_condition
            case 'nonboundary_edge_fixed':
                center_edge = torch.exp(-((curvilinear_coords[...,0:1] - 0.5 * self.curvilinear_space.xi__1_max) ** 2)/self.boundary_support)
                deformations = deformations * (1 - center_edge)
            case 'top_bottom_rims_compression':                
                bottom_rim = torch.exp(-(curvilinear_coords[...,1:2] ** 2)/self.boundary_support)
                top_rim = torch.exp(-((curvilinear_coords[...,1:2] - self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)
                #rim_displacement = torch.cat([torch.zeros_like(temporal_coords), 0.075 * temporal_coords, torch.zeros_like(temporal_coords)], dim=2)
                rim_displacement = torch.cat([torch.zeros_like(temporal_coords), 0.075 * torch.ones_like(temporal_coords), torch.zeros_like(temporal_coords)], dim=2)
                #deformations = deformations * (1 - bottom_rim) * (1 - top_rim) * initial_condition - rim_displacement * top_rim + rim_displacement * bottom_rim
                deformations = deformations * (1 - bottom_rim) * (1 - top_rim) - rim_displacement * top_rim + rim_displacement * bottom_rim
            case 'top_bottom_rims_torsion':
                bottom_rim = torch.exp(-(curvilinear_coords[...,1][...,None] ** 2)/self.boundary_support)
                top_rim = torch.exp(-((curvilinear_coords[...,1][...,None] - self.theta__2_scale) ** 2)/self.boundary_support)
                R = 0.25
                rotation = 3 * math.pi / 4 
                top_rim_displacement = torch.cat([R * (torch.cos(curvilinear_coords[...,0][...,None] + (temporal_coords ** 2) * rotation) - torch.cos(curvilinear_coords[...,0][...,None])), R * (torch.sin(curvilinear_coords[...,0][...,None] + (temporal_coords ** 2) * rotation) - torch.sin(curvilinear_coords[...,0][...,None])), torch.zeros_like(temporal_coords)], dim=2)
                bottom_rim_displacement = torch.cat([R * (torch.cos(curvilinear_coords[...,0][...,None] - (temporal_coords ** 2) * rotation) - torch.cos(curvilinear_coords[...,0][...,None])), R * (torch.sin(curvilinear_coords[...,0][...,None] - (temporal_coords ** 2) * rotation) - torch.sin(curvilinear_coords[...,0][...,None])), torch.zeros_like(temporal_coords)], dim=2)            
                #output = output * (1 - bottom_rim) * (1 - top_rim) * initial_condition + top_rim_displacement * top_rim + bottom_rim_displacement * bottom_rim
                output = (output * (1 - bottom_rim) * (1 - top_rim) + top_rim_displacement * top_rim + bottom_rim_displacement * bottom_rim) * initial_condition
                #output = (output * (1 - bottom_rim) * (1 - top_rim) + top_rim_displacement * top_rim + bottom_rim_displacement * bottom_rim) * initial_condition #Leads to inward motion of cylinder at time step 7/10                
            case 'top_rim_fixed':
                top_rim = torch.exp(-((curvilinear_coords[...,1:2] - self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)
                deformations = deformations * (1 - top_rim) #* initial_condition
            case 'top_rim_torsion':
                top_rim = torch.exp(-((curvilinear_coords[...,1:2] - self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)
                R_top = 0.2
                top_rim_displacement = torch.cat([R_top * (torch.cos(curvilinear_coords[...,0:1] + temporal_coords * math.pi/2) - torch.cos(curvilinear_coords[...,0:1])), torch.zeros_like(temporal_coords), R_top * (torch.sin(curvilinear_coords[...,0:1] + temporal_coords * math.pi/2) - torch.sin(curvilinear_coords[...,0:1]))], dim=2)
                deformations = deformations * (1 - top_rim) * initial_condition + top_rim_displacement * top_rim            
            case 'mesh_vertices':
                for i in range(self.boundary_curvilinear_coords.shape[0]):
                    boundary_point = torch.exp(-((curvilinear_coords[...,0:1] - self.boundary_curvilinear_coords[i][0]) ** 2 + (curvilinear_coords[...,1:2] - self.boundary_curvilinear_coords[i][1]) ** 2)/self.boundary_support)
                    deformations = deformations * (1 - boundary_point)
            case _:
                raise ValueError(f'Unknown boundary condition: {self.boundary_condition_name}')
        return deformations