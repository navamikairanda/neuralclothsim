import torch

boundary_support = 0.01

def apply_boundary(deformations, curvilinear_coords, boundary_condition_name, xi__1_max, xi__2_max):
    match boundary_condition_name:
        case 'top_left_fixed':
            top_left_corner = torch.exp(-(curvilinear_coords[...,0:1] ** 2 + (curvilinear_coords[...,1:2] - xi__2_max) ** 2)/boundary_support)
            deformations = deformations * (1 - top_left_corner)
        case _:
            raise ValueError(f"Unknown boundary condition: {boundary_condition_name}")
    return deformations