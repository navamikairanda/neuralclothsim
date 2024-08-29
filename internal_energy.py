import torch
from torch.nn.functional import normalize
from diff_operators import jacobian
from material import Material, LinearMaterial, NonLinearMaterial, MaterialOrthotropy
from plot_helper import get_plot_single_tensor
from reference_geometry import ReferenceGeometry
from strain import Strain, compute_strain
                
def compute_nonlinear_internal_energy(strain: Strain, material: NonLinearMaterial, material_directions: MaterialOrthotropy, ref_geometry: ReferenceGeometry, tb_writer, i, xi__3: float) -> torch.Tensor:
    # _ob are the strains in the original basis
    # Eq. (22)
    E11_ob = strain.epsilon_1_1 + xi__3 * strain.kappa_1_1
    E12_ob = strain.epsilon_1_2 + xi__3 * strain.kappa_1_2
    E22_ob = strain.epsilon_2_2 + xi__3 * strain.kappa_2_2
    g__1__1, g__1__2, g__2__1, g__2__2 = ref_geometry.contravariant_base_vectors(xi__3)
    
    E = torch.einsum('ij,ijkl->ijkl', E11_ob, g__1__1) + torch.einsum('ij,ijkl->ijkl', E12_ob, g__1__2) + torch.einsum('ij,ijkl->ijkl', E12_ob, g__2__1) + torch.einsum('ij,ijkl->ijkl', E22_ob, g__2__2)
    E11 = torch.einsum('ijk,ijkl,ijl->ij', material_directions.d_1, E, material_directions.d_1)
    E12 = torch.einsum('ijk,ijkl,ijl->ij', material_directions.d_1, E, material_directions.d_2)
    E22 = torch.einsum('ijk,ijkl,ijl->ij', material_directions.d_2, E, material_directions.d_2)
    
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

def compute_linear_internal_energy(strain: Strain, material: LinearMaterial, ref_geometry: ReferenceGeometry) -> torch.Tensor:
    H__1111, H__1112, H__1122, H__1212, H__1222, H__2222 = ref_geometry.elastic_tensor(material.poissons_ratio)
        
    n__1__1 = H__1111 * strain.epsilon_1_1 + 2 * H__1112 * strain.epsilon_1_2 + H__1122 * strain.epsilon_2_2
    n__1__2 = H__1112 * strain.epsilon_1_1 + 2 * H__1212 * strain.epsilon_1_2 + H__1222 * strain.epsilon_2_2
    n__2__1 = n__1__2
    n__2__2 = H__1122 * strain.epsilon_1_1 + 2 * H__1222 * strain.epsilon_1_2 + H__2222 * strain.epsilon_2_2

    m__1__1 = H__1111 * strain.kappa_1_1 + 2 * H__1112 * strain.kappa_1_2 + H__1122 * strain.kappa_2_2
    m__1__2 = H__1112 * strain.kappa_1_1 + 2 * H__1212 * strain.kappa_1_2 + H__1222 * strain.kappa_2_2
    m__2__1 = m__1__2
    m__2__2 = H__1122 * strain.kappa_1_1 + 2 * H__1222 * strain.kappa_1_2 + H__2222 * strain.kappa_2_2
    
    hyperelastic_strain_energy = 0.5 * (material.D * (strain.epsilon_1_1 * n__1__1 + strain.epsilon_1_2 * n__1__2 + strain.epsilon_1_2 * n__2__1 + strain.epsilon_2_2 * n__2__2) + material.B * (strain.kappa_1_1 * m__1__1 + strain.kappa_1_2 * m__1__2 + strain.kappa_1_2 * m__2__1 + strain.kappa_2_2 * m__2__2))
    
    return hyperelastic_strain_energy

def compute_energy(deformations: torch.Tensor, ref_geometry: ReferenceGeometry, material: Material, external_load: torch.Tensor, temporal_coords: torch.Tensor, tb_writer, i: int) -> torch.Tensor:
    deformations_local = torch.einsum('ijkl,ijl->ijk', ref_geometry.cartesian_coord_2_covariant, deformations)
    strain, normal_difference = compute_strain(deformations_local, ref_geometry, tb_writer, i)
    
    if isinstance(material, LinearMaterial):        
        hyperelastic_strain_energy_mid = compute_linear_internal_energy(strain, material, ref_geometry)
        external_energy_mid = torch.einsum('ijk,ijk->ij', external_load, deformations)
        velocity = jacobian(deformations, temporal_coords)[0]
        kinetic_energy = 0.5 * material.mass_area_density * torch.einsum('ijkl,ijkl->ij', velocity, velocity)
        mechanical_energy = (hyperelastic_strain_energy_mid - external_energy_mid) * torch.sqrt(ref_geometry.a) #+ kinetic_energy
    elif isinstance(material, NonLinearMaterial):
        d_1 = normalize((ref_geometry.a_1), dim=2)
        d_2 = torch.linalg.cross(ref_geometry.a_3, d_1)
        material_directions = MaterialOrthotropy(d_1, d_2)
        hyperelastic_strain_energy_top = compute_nonlinear_internal_energy(strain, material, material_directions, ref_geometry, tb_writer, i, -0.5 * material.thickness)
        hyperelastic_strain_energy_mid = compute_nonlinear_internal_energy(strain, material, material_directions, ref_geometry, tb_writer, i, 0.)
        hyperelastic_strain_energy_bottom = compute_nonlinear_internal_energy(strain, material, material_directions, ref_geometry, tb_writer, i, 0.5 * material.thickness)

        #hyperelastic_strain_energy = (hyperelastic_strain_energy_top + hyperelastic_strain_energy_mid + hyperelastic_strain_energy_bottom) / 3.        
        external_energy_mid = torch.einsum('ijk,ijk->ij', external_load, deformations)
        
        #velocity = jacobian(deformations_mid, temporal_coords)[0]
        #kinetic_energy = 0.5 * material.mass_area_density * torch.einsum('ijkl,ijkl->ij', velocity, velocity)        
        #mechanical_energy = ((hyperelastic_strain_energy_top + kinetic_energy - external_energy_top) + 4 * (hyperelastic_strain_energy_mid + kinetic_energy - external_energy_mid) + (external_energy_bottom + kinetic_energy - hyperelastic_strain_energy_bottom)) * torch.sqrt(ref_geometry.a) / 6.        
        deformations_top = deformations - 0.5 * material.thickness * normal_difference
        deformations_bottom = deformations + 0.5 * material.thickness * normal_difference
        external_energy_top = torch.einsum('ijk,ijk->ij', external_load, deformations_top)
        external_energy_bottom = torch.einsum('ijk,ijk->ij', external_load, deformations_bottom)
        mechanical_energy = ((hyperelastic_strain_energy_top - external_energy_top) + 4 * (hyperelastic_strain_energy_mid - external_energy_mid) + (external_energy_bottom - hyperelastic_strain_energy_bottom)) * torch.sqrt(ref_geometry.a) / 6.        
        #mechanical_energy = (hyperelastic_strain_energy_mid - external_energy_mid) * torch.sqrt(ref_geometry.a)
    if not i % 200:
        tb_writer.add_figure(f'hyperelastic_strain_energy', get_plot_single_tensor(hyperelastic_strain_energy_mid[0,:ref_geometry.spatial_sidelen**2], ref_geometry.spatial_sidelen), i)
        tb_writer.add_histogram('param/hyperelastic_strain_energy', hyperelastic_strain_energy_mid, i)
        #print(f'min_hyperelastic_strain_energy: {hyperelastic_strain_energy.min()}')    
    return mechanical_energy