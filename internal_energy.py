import torch
from torch.nn.functional import normalize
from diff_operators import jacobian
from material import Material, LinearMaterial, NonLinearMaterial
from plot_helper import get_plot_grid_tensor, get_plot_single_tensor
from reference_geometry import ReferenceGeometry

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

def compute_strain(deformations: torch.Tensor, ref_geometry: ReferenceGeometry, nonlinear=True): 
    
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
    
    #return epsilon_1_1, epsilon_1_2, epsilon_2_2, kappa_1_1, kappa_1_2, kappa_2_2
    w_1 = -phi_1_3 + phi_1__1 * phi_1_3 + phi_1__2 * phi_2_3
    w_2 = -phi_2_3 + phi_2__1 * phi_1_3 + phi_2__2 * phi_2_3
    w_3 = 0.5 * (phi_1_3 * phi_3__1 + phi_2_3 * phi_3__2)
    w = torch.einsum('ij,ijk->ijk', w_1, ref_geometry.a__1) + torch.einsum('ij,ijk->ijk', w_2, ref_geometry.a__2) + torch.einsum('ij,ijk->ijk', w_3, ref_geometry.a_3)
    return epsilon_1_1, epsilon_1_2, epsilon_2_2, kappa_1_1, kappa_1_2, kappa_2_2, w

@torch.no_grad()
def elastic_tensor(ref_geometry: ReferenceGeometry, poissons_ratio: float):
    H__1111 = poissons_ratio * ref_geometry.a__1__1 * ref_geometry.a__1__1 + 0.5 * (1 - poissons_ratio) * (ref_geometry.a__1__1 * ref_geometry.a__1__1 + ref_geometry.a__1__1 * ref_geometry.a__1__1)
    H__1112 = poissons_ratio * ref_geometry.a__1__1 * ref_geometry.a__1__2 + 0.5 * (1 - poissons_ratio) * (ref_geometry.a__1__1 * ref_geometry.a__1__2 + ref_geometry.a__1__2 * ref_geometry.a__1__1)
    H__1122 = poissons_ratio * ref_geometry.a__1__1 * ref_geometry.a__2__2 + 0.5 * (1 - poissons_ratio) * (ref_geometry.a__1__2 * ref_geometry.a__1__2 + ref_geometry.a__1__2 * ref_geometry.a__1__2)
    H__1212 = poissons_ratio * ref_geometry.a__1__2 * ref_geometry.a__1__2 + 0.5 * (1 - poissons_ratio) * (ref_geometry.a__1__1 * ref_geometry.a__2__2 + ref_geometry.a__1__2 * ref_geometry.a__2__1)
    H__1222 = poissons_ratio * ref_geometry.a__1__2 * ref_geometry.a__2__2 + 0.5 * (1 - poissons_ratio) * (ref_geometry.a__1__2 * ref_geometry.a__2__2 + ref_geometry.a__1__2 * ref_geometry.a__2__2)
    H__2222 = poissons_ratio * ref_geometry.a__2__2 * ref_geometry.a__2__2 + 0.5 * (1 - poissons_ratio) * (ref_geometry.a__2__2 * ref_geometry.a__2__2 + ref_geometry.a__2__2 * ref_geometry.a__2__2)
    return H__1111, H__1112, H__1122, H__1212, H__1222, H__2222

def contravariant_base_vectors(ref_geometry, xi__3):
    g_1_1 = ref_geometry.a_1_1.repeat(1, ref_geometry.temporal_sidelen) - 2 * xi__3 * ref_geometry.b_1_1
    g_1_2 = ref_geometry.a_1_2.repeat(1, ref_geometry.temporal_sidelen) - 2 * xi__3 * ref_geometry.b_1_2
    g_2_2 = ref_geometry.a_2_2.repeat(1, ref_geometry.temporal_sidelen) - 2 * xi__3 * ref_geometry.b_2_2
    
    #a_3pd = jacobian(ref_geometry.a_3, ref_geometry.curvilinear_coords)[0]
    g_1 = ref_geometry.a_1 + xi__3 * ref_geometry.a_3pd[...,0].repeat(1, ref_geometry.temporal_sidelen, 1)
    g_2 = ref_geometry.a_2 + xi__3 * ref_geometry.a_3pd[...,1].repeat(1, ref_geometry.temporal_sidelen, 1)
    g_covariant_matrix = torch.stack([torch.stack([g_1_1, g_1_2], dim=2), torch.stack([g_1_2, g_2_2], dim=2)], dim=2) 
    g_contravariant_matrix = torch.linalg.inv(g_covariant_matrix)
    g__1 = torch.einsum('ij,ijk->ijk', g_contravariant_matrix[...,0,0], g_1) + torch.einsum('ij,ijk->ijk', g_contravariant_matrix[...,0,1], g_2)
    g__2 = torch.einsum('ij,ijk->ijk', g_contravariant_matrix[...,1,0], g_1) + torch.einsum('ij,ijk->ijk', g_contravariant_matrix[...,1,1], g_2)

    g__1__1 = torch.einsum('ijk,ijl->ijkl', g__1, g__1)
    g__1__2 = torch.einsum('ijk,ijl->ijkl', g__1, g__2)
    g__2__1 = torch.einsum('ijk,ijl->ijkl', g__2, g__1)
    g__2__2 = torch.einsum('ijk,ijl->ijkl', g__2, g__2)
    return g__1__1, g__1__2, g__2__1, g__2__2
            
def compute_nonlinear_internal_energy(epsilon_1_1: torch.Tensor, epsilon_1_2: torch.Tensor, epsilon_2_2: torch.Tensor, kappa_1_1: torch.Tensor, kappa_1_2: torch.Tensor, kappa_2_2: torch.Tensor, material: torch.Tensor, material_direction_1, material_direction_2, ref_geometry, tb_writer, i, xi__3):
    # _ob are the strains in the original basis
    E11_ob = epsilon_1_1 + xi__3 * kappa_1_1
    E12_ob = epsilon_1_2 + xi__3 * kappa_1_2
    E22_ob = epsilon_2_2 + xi__3 * kappa_2_2
    g__1__1, g__1__2, g__2__1, g__2__2 = contravariant_base_vectors(ref_geometry, xi__3)
    E = torch.einsum('ij,ijkl->ijkl', E11_ob, g__1__1) + torch.einsum('ij,ijkl->ijkl', E12_ob, g__1__2) + torch.einsum('ij,ijkl->ijkl', E12_ob, g__2__1) + torch.einsum('ij,ijkl->ijkl', E22_ob, g__2__2)
    E11 = torch.einsum('ijk,ijkl,ijl->ij', material_direction_1, E, material_direction_1)
    E12 = torch.einsum('ijk,ijkl,ijl->ij', material_direction_1, E, material_direction_2)
    E22 = torch.einsum('ijk,ijkl,ijl->ij', material_direction_2, E, material_direction_2)
    '''
    E11 = material_direction_1[...,0] ** 2 * E11_ob + 2 * material_direction_1[...,0] * material_direction_1[...,1] * E12_ob + material_direction_1[...,1] ** 2 * E22_ob
    E12 = material_direction_1[...,0] * material_direction_2[...,0] * E11_ob + (material_direction_1[...,0] * material_direction_2[...,1] + material_direction_1[...,1] * material_direction_2[...,0]) * E12_ob + material_direction_1[...,1] * material_direction_2[...,1] * E22_ob
    E22 = material_direction_2[...,0] ** 2 * E11_ob + 2 * material_direction_2[...,0] * material_direction_2[...,1] * E12_ob + material_direction_2[...,1] ** 2 * E22_ob
    '''
    #E11, E12, E22 = E11_ob, E12_ob, E22_ob    
    if material.StVK:        
        # St. Venant-Kirchhoff model [Basar 2000]
        hyperelastic_strain_energy = material.a11 * 0.5 * E11 ** 2 + material.a12 * E11 * E22 + material.a22 * 0.5 * E22 ** 2 + material.G12 * E12 ** 2                  
    else:
        E11_valid = torch.logical_and(E11 > material.E_11_min, E11 < material.E_11_max)
        E12_valid = torch.logical_and(E12 > material.E_12_min, E12 < material.E_12_max)
        E22_valid = torch.logical_and(E22 > material.E_22_min, E22 < material.E_22_max)
             
        E11_clamped = torch.clamp(E11, material.E_11_min, material.E_11_max)
        E12_clamped = torch.clamp(E12, material.E_12_min, material.E_12_max)
        E22_clamped = torch.clamp(E22, material.E_22_min, material.E_22_max)
        
        # TODO precompute the square of the strains, or the derivatives of the strains 
        E11_22_clamped = E11_clamped * E22_clamped    
 
        hyperelastic_strain_energy = material.a11 * 0.5 * material.compute_eta(0, E11_clamped ** 2) + material.a12 * material.compute_eta(1, E11_22_clamped) + material.a22 * 0.5 * material.compute_eta(2, E22_clamped ** 2) + material.G12 * material.compute_eta(3, E12_clamped ** 2)

        eta_first_derivative_3_E12_12_clamped = material.compute_eta_first_derivative(3, E12_clamped ** 2)
        eta_first_derivative_2_E22_22_clamped = material.compute_eta_first_derivative(2, E22_clamped ** 2)
        eta_first_derivative_1_E11_22_clamped = material.compute_eta_first_derivative(1, E11_clamped * E22_clamped)
        eta_first_derivative_0_E11_11_clamped = material.compute_eta_first_derivative(0, E11_clamped ** 2)
        eta_second_derivative_1_E11_22_clamped = material.compute_eta_second_derivative(1, E11_clamped * E22_clamped)
        
        hyperelastic_strain_energy += ~E12_valid * (2 * material.G12 * E12_clamped * eta_first_derivative_3_E12_12_clamped * (E12 - E12_clamped) + 0.5 * (2 * material.G12 * eta_first_derivative_3_E12_12_clamped + 4 * material.G12 * E12_clamped ** 2 * material.compute_eta_second_derivative(3, E12_clamped ** 2)) * (E12 - E12_clamped) ** 2)
        
        hyperelastic_strain_energy += ~E11_valid * ((material.a11 * E11_clamped * eta_first_derivative_0_E11_11_clamped + material.a12 * E22_clamped * eta_first_derivative_1_E11_22_clamped) * (E11 - E11_clamped) + 0.5 * (material.a11 * eta_first_derivative_0_E11_11_clamped + 2 * material.a11 * E11_clamped ** 2 * material.compute_eta_second_derivative(0, E11_clamped ** 2) + material.a12 * E22_clamped ** 2 * eta_second_derivative_1_E11_22_clamped) * (E11 - E11_clamped) ** 2)
        
        hyperelastic_strain_energy += ~E22_valid * ((material.a22 * E22_clamped * eta_first_derivative_2_E22_22_clamped + material.a12 * E11_clamped * eta_first_derivative_1_E11_22_clamped) * (E22 - E22_clamped) + 0.5 * (material.a22 * eta_first_derivative_2_E22_22_clamped + 2 * material.a22 * E22_clamped ** 2 * material.compute_eta_second_derivative(2, E22_clamped ** 2) + material.a12 * E11_clamped ** 2 * eta_second_derivative_1_E11_22_clamped) * (E22 - E22_clamped) ** 2)
        
        #Where both are not valid, the energy is zero
        hyperelastic_strain_energy += (~E11_valid * ~E22_valid) * (material.a12 * eta_first_derivative_1_E11_22_clamped + material.a12 * E11_clamped * E22_clamped * eta_second_derivative_1_E11_22_clamped) * (E11 - E11_clamped) * (E22 - E22_clamped)
        
        if not i % 200:
            tb_writer.add_histogram('param/E11_valid', E11_valid, i)
            tb_writer.add_histogram('param/E12_valid', E12_valid, i)
            tb_writer.add_histogram('param/E22_valid', E22_valid, i)
            tb_writer.add_histogram('param/E11_min_valid', E11 > material.E_11_min, i)
            tb_writer.add_histogram('param/E11_max_valid', E11 < material.E_11_max, i)
            tb_writer.add_histogram('param/E12_min_valid', E12 > material.E_12_min, i)
            tb_writer.add_histogram('param/E12_max_valid', E12 < material.E_12_max, i)
            tb_writer.add_histogram('param/E22_min_valid', E22 > material.E_22_min, i)
            tb_writer.add_histogram('param/E22_max_valid', E22 < material.E_22_max, i)
        
    if not i % 200:
        tb_writer.add_histogram('param/E11', E11, i)
        tb_writer.add_histogram('param/E12', E12, i)
        tb_writer.add_histogram('param/E22', E22, i)
    
    return hyperelastic_strain_energy

def compute_linear_internal_energy(epsilon_1_1: torch.Tensor, epsilon_1_2: torch.Tensor, epsilon_2_2: torch.Tensor, kappa_1_1: torch.Tensor, kappa_1_2: torch.Tensor, kappa_2_2: torch.Tensor, material: LinearMaterial, ref_geometry: ReferenceGeometry) -> torch.Tensor:
    epsilon_2_1 = epsilon_1_2
    kappa_2_1 = kappa_1_2
    H__1111, H__1112, H__1122, H__1212, H__1222, H__2222 = elastic_tensor(ref_geometry, material.poissons_ratio)
        
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

def compute_energy(deformations: torch.Tensor, ref_geometry: ReferenceGeometry, material: Material, external_load: torch.Tensor, temporal_coords: torch.Tensor, tb_writer, i: int) -> torch.Tensor:
    deformations_local = torch.einsum('ijkl,ijl->ijk', ref_geometry.cartesian_coord_2_covariant, deformations)
    epsilon_1_1, epsilon_1_2, epsilon_2_2, kappa_1_1, kappa_1_2, kappa_2_2, w = compute_strain(deformations_local, ref_geometry)
    
    if not i % 200:
        tb_writer.add_figure(f'membrane_strain', get_plot_grid_tensor(epsilon_1_1[0,:ref_geometry.spatial_sidelen**2], epsilon_1_2[0,:ref_geometry.spatial_sidelen**2], epsilon_1_2[0,:ref_geometry.spatial_sidelen**2], epsilon_2_2[0,:ref_geometry.spatial_sidelen**2], ref_geometry.spatial_sidelen), i)
        tb_writer.add_figure(f'bending_strain', get_plot_grid_tensor(kappa_1_1[0,:ref_geometry.spatial_sidelen**2], kappa_1_2[0,:ref_geometry.spatial_sidelen**2], kappa_1_2[0,:ref_geometry.spatial_sidelen**2], kappa_2_2[0,:ref_geometry.spatial_sidelen**2], ref_geometry.spatial_sidelen), i)
    
    if isinstance(material, LinearMaterial):        
        hyperelastic_strain_energy_mid = compute_linear_internal_energy(epsilon_1_1, epsilon_1_2, epsilon_2_2, kappa_1_1, kappa_1_2, kappa_2_2, material, ref_geometry)
        external_energy_mid = torch.einsum('ijk,ijk->ij', external_load, deformations)
        velocity = jacobian(deformations, temporal_coords)[0]
        kinetic_energy = 0.5 * material.mass_area_density * torch.einsum('ijkl,ijkl->ij', velocity, velocity)
        mechanical_energy = (hyperelastic_strain_energy_mid - external_energy_mid) * torch.sqrt(ref_geometry.a) #+ kinetic_energy
    elif isinstance(material, NonLinearMaterial):
        material_direction_1 = normalize((ref_geometry.a_1), dim=2)
        material_direction_2 = torch.linalg.cross(ref_geometry.a_3, material_direction_1)
        #material_direction_3 = ref_geometry.a_3        
        hyperelastic_strain_energy_top = compute_nonlinear_internal_energy(epsilon_1_1, epsilon_1_2, epsilon_2_2, kappa_1_1, kappa_1_2, kappa_2_2, material, material_direction_1, material_direction_2, ref_geometry, tb_writer, i, -0.5 * material.thickness)
        hyperelastic_strain_energy_mid = compute_nonlinear_internal_energy(epsilon_1_1, epsilon_1_2, epsilon_2_2, kappa_1_1, kappa_1_2, kappa_2_2, material, material_direction_1, material_direction_2, ref_geometry, tb_writer, i, 0.)
        hyperelastic_strain_energy_bottom = compute_nonlinear_internal_energy(epsilon_1_1, epsilon_1_2, epsilon_2_2, kappa_1_1, kappa_1_2, kappa_2_2, material, material_direction_1, material_direction_2, ref_geometry, tb_writer, i, 0.5 * material.thickness)

        #hyperelastic_strain_energy = (hyperelastic_strain_energy_top + hyperelastic_strain_energy_mid + hyperelastic_strain_energy_bottom) / 3.        
        deformations_mid = deformations
        external_energy_mid = torch.einsum('ijk,ijk->ij', external_load, deformations_mid)
        
        #velocity = jacobian(deformations_mid, temporal_coords)[0]
        #kinetic_energy = 0.5 * material.mass_area_density * torch.einsum('ijkl,ijkl->ij', velocity, velocity)        
        #mechanical_energy = ((hyperelastic_strain_energy_top + kinetic_energy - external_energy_top) + 4 * (hyperelastic_strain_energy_mid + kinetic_energy - external_energy_mid) + (external_energy_bottom + kinetic_energy - hyperelastic_strain_energy_bottom)) * torch.sqrt(ref_geometry.a) / 6.        
        deformations_top = deformations - 0.5 * material.thickness * w
        deformations_bottom = deformations + 0.5 * material.thickness * w
        external_energy_top = torch.einsum('ijk,ijk->ij', external_load, deformations_top)
        external_energy_bottom = torch.einsum('ijk,ijk->ij', external_load, deformations_bottom)
        mechanical_energy = ((hyperelastic_strain_energy_top - external_energy_top) + 4 * (hyperelastic_strain_energy_mid - external_energy_mid) + (external_energy_bottom - hyperelastic_strain_energy_bottom)) * torch.sqrt(ref_geometry.a) / 6.        
        #mechanical_energy = (hyperelastic_strain_energy_mid - external_energy_mid) * torch.sqrt(ref_geometry.a)
    if not i % 200:
        tb_writer.add_figure(f'hyperelastic_strain_energy', get_plot_single_tensor(hyperelastic_strain_energy_mid[0,:ref_geometry.spatial_sidelen**2], ref_geometry.spatial_sidelen), i)
        tb_writer.add_histogram('param/hyperelastic_strain_energy', hyperelastic_strain_energy_mid, i)
        #print(f'min_hyperelastic_strain_energy: {hyperelastic_strain_energy.min()}')    
    return mechanical_energy