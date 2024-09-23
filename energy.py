import torch
from torch.nn.functional import normalize
import utils.tb as tb
from utils.diff_operators import jacobian
from utils.config_parser import device
from material import Material, LinearMaterial, NonLinearMaterial, MaterialOrthotropy
from utils.plot import get_plot_single_tensor
from reference_geometry import ReferenceGeometry
from strain import compute_strain                

class Energy:
    def __init__(self, ref_geometry: ReferenceGeometry, material: Material, gravity_acceleration: list, i_debug: int):
        self.ref_geometry = ref_geometry
        self.material = material
        external_load = torch.tensor(gravity_acceleration, device=device) * material.mass_area_density
        self.external_load = external_load.expand(1, ref_geometry.n_spatial_samples, 3)
        self.i_debug = i_debug
        
    def __call__(self, deformations: torch.Tensor, i: int) -> torch.Tensor:   
        strain, normal_difference = compute_strain(deformations, self.ref_geometry, i, self.i_debug)
        
        if isinstance(self.material, LinearMaterial):        
            hyperelastic_strain_energy_mid = self.material.compute_internal_energy(strain)
            external_energy_mid = torch.einsum('ijk,ijk->ij', self.external_load, deformations)
            mechanical_energy = (hyperelastic_strain_energy_mid - external_energy_mid)
            mechanical_energy = mechanical_energy * torch.sqrt(self.ref_geometry.a)
            
        elif isinstance(self.material, NonLinearMaterial):
            d_1 = normalize((self.ref_geometry.a_1), dim=2)
            d_2 = torch.linalg.cross(self.ref_geometry.a_3, d_1)
            material_directions = MaterialOrthotropy(d_1, d_2)
            hyperelastic_strain_energy_top = self.material.compute_internal_energy(strain, material_directions, i, -0.5 * self.material.thickness)
            hyperelastic_strain_energy_mid = self.material.compute_internal_energy(strain, material_directions, i, 0.)
            hyperelastic_strain_energy_bottom = self.material.compute_internal_energy(strain, material_directions, i, 0.5 * self.material.thickness)                      
            
            external_energy_top = torch.einsum('ijk,ijk->ij', self.external_load, deformations - 0.5 * self.material.thickness * normal_difference)
            external_energy_mid = torch.einsum('ijk,ijk->ij', self.external_load, deformations)
            external_energy_bottom = torch.einsum('ijk,ijk->ij', self.external_load, deformations + 0.5 * self.material.thickness * normal_difference)
            mechanical_energy = (hyperelastic_strain_energy_top - external_energy_top) + 4 * (hyperelastic_strain_energy_mid - external_energy_mid) + (external_energy_bottom - hyperelastic_strain_energy_bottom)
            mechanical_energy = mechanical_energy * torch.sqrt(self.ref_geometry.a) / 6.
        if not i % self.i_debug and tb.writer:
            tb.writer.add_histogram('param/hyperelastic_strain_energy', hyperelastic_strain_energy_mid, i) 
            if self.ref_geometry.reference_midsurface.reference_geometry_name != 'mesh':
                tb.writer.add_figure(f'hyperelastic_strain_energy', get_plot_single_tensor(hyperelastic_strain_energy_mid[0,-self.ref_geometry.n_spatial_samples:]), i)
        return mechanical_energy