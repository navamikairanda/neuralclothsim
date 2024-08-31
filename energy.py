import torch
from torch.nn.functional import normalize
from diff_operators import jacobian
from material import Material, LinearMaterial, NonLinearMaterial, MaterialOrthotropy
from plot_helper import get_plot_single_tensor
from reference_geometry import ReferenceGeometry
from strain import compute_strain                

def compute_energy(deformations: torch.Tensor, ref_geometry: ReferenceGeometry, material: Material, external_load: torch.Tensor, temporal_coords: torch.Tensor, tb_writer, i: int) -> torch.Tensor:   
    strain, normal_difference = compute_strain(deformations, ref_geometry, tb_writer, i)
    
    if isinstance(material, LinearMaterial):        
        hyperelastic_strain_energy_mid = material.compute_internal_energy(strain, ref_geometry)
        external_energy_mid = torch.einsum('ijk,ijk->ij', external_load, deformations)
        velocity = jacobian(deformations, temporal_coords)[0]
        kinetic_energy = 0.5 * material.mass_area_density * torch.einsum('ijkl,ijkl->ij', velocity, velocity)
        mechanical_energy = (hyperelastic_strain_energy_mid - external_energy_mid) * torch.sqrt(ref_geometry.a) #+ kinetic_energy
        
    elif isinstance(material, NonLinearMaterial):
        d_1 = normalize((ref_geometry.a_1), dim=2)
        d_2 = torch.linalg.cross(ref_geometry.a_3, d_1)
        material_directions = MaterialOrthotropy(d_1, d_2)
        hyperelastic_strain_energy_top = material.compute_internal_energy(strain, ref_geometry, material_directions, tb_writer, i, -0.5 * material.thickness)
        hyperelastic_strain_energy_mid = material.compute_internal_energy(strain, ref_geometry, material_directions, tb_writer, i, 0.)
        hyperelastic_strain_energy_bottom = material.compute_internal_energy(strain, ref_geometry, material_directions, tb_writer, i, 0.5 * material.thickness)                      
        
        #velocity = jacobian(deformations_mid, temporal_coords)[0]
        #kinetic_energy = 0.5 * material.mass_area_density * torch.einsum('ijkl,ijkl->ij', velocity, velocity)        
        #mechanical_energy = ((hyperelastic_strain_energy_top + kinetic_energy - external_energy_top) + 4 * (hyperelastic_strain_energy_mid + kinetic_energy - external_energy_mid) + (external_energy_bottom + kinetic_energy - hyperelastic_strain_energy_bottom)) * torch.sqrt(ref_geometry.a) / 6.        
        deformations_top = deformations - 0.5 * material.thickness * normal_difference
        deformations_bottom = deformations + 0.5 * material.thickness * normal_difference
        external_energy_top = torch.einsum('ijk,ijk->ij', external_load, deformations_top)
        external_energy_mid = torch.einsum('ijk,ijk->ij', external_load, deformations)
        external_energy_bottom = torch.einsum('ijk,ijk->ij', external_load, deformations_bottom)
        mechanical_energy = ((hyperelastic_strain_energy_top - external_energy_top) + 4 * (hyperelastic_strain_energy_mid - external_energy_mid) + (external_energy_bottom - hyperelastic_strain_energy_bottom)) * torch.sqrt(ref_geometry.a) / 6.        
    if not i % 200:
        tb_writer.add_figure(f'hyperelastic_strain_energy', get_plot_single_tensor(hyperelastic_strain_energy_mid[0,:ref_geometry.spatial_sidelen**2], ref_geometry.spatial_sidelen), i)
        tb_writer.add_histogram('param/hyperelastic_strain_energy', hyperelastic_strain_energy_mid, i)          
    return mechanical_energy