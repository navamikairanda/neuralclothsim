import torch
from typing import NamedTuple
from diff_operators import jacobian
from reference_geometry import ReferenceGeometry
from plot_helper import get_plot_grid_tensor

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

def compute_strain(deformations: torch.Tensor, ref_geometry: ReferenceGeometry, tb_writer, i: int, nonlinear_strain=True): 
    
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
    
    # Eq. (18), Theorem B.3, Eq. (51)
    w_1 = -phi_1_3 + phi_1__1 * phi_1_3 + phi_1__2 * phi_2_3
    w_2 = -phi_2_3 + phi_2__1 * phi_1_3 + phi_2__2 * phi_2_3
    w_3 = 0.5 * (phi_1_3 * phi_3__1 + phi_2_3 * phi_3__2)
    normal_difference = torch.einsum('ij,ijk->ijk', w_1, ref_geometry.a__1) + torch.einsum('ij,ijk->ijk', w_2, ref_geometry.a__2) + torch.einsum('ij,ijk->ijk', w_3, ref_geometry.a_3)
    
    if not i % 200:
        tb_writer.add_figure(f'membrane_strain', get_plot_grid_tensor(epsilon_1_1[0,:ref_geometry.spatial_sidelen**2], epsilon_1_2[0,:ref_geometry.spatial_sidelen**2], epsilon_1_2[0,:ref_geometry.spatial_sidelen**2], epsilon_2_2[0,:ref_geometry.spatial_sidelen**2], ref_geometry.spatial_sidelen), i)
        tb_writer.add_figure(f'bending_strain', get_plot_grid_tensor(kappa_1_1[0,:ref_geometry.spatial_sidelen**2], kappa_1_2[0,:ref_geometry.spatial_sidelen**2], kappa_1_2[0,:ref_geometry.spatial_sidelen**2], kappa_2_2[0,:ref_geometry.spatial_sidelen**2], ref_geometry.spatial_sidelen), i)
    
    return Strain(epsilon_1_1, epsilon_1_2, epsilon_2_2, kappa_1_1, kappa_1_2, kappa_2_2), normal_difference
