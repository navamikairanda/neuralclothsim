import torch
from typing import NamedTuple
import tb
from strain import Strain
from reference_geometry import ReferenceGeometry
    
class Material():
    def __init__(self, mass_area_density: float, thickness: float, ref_geometry: ReferenceGeometry):
        self.thickness = thickness
        self.mass_area_density = mass_area_density
        self.ref_geometry = ref_geometry
    
    def compute_internal_energy(self, strain: Strain) -> torch.Tensor:
        raise NotImplementedError
        
class LinearMaterial(Material):
    def __init__(self, args, ref_geometry):
        super().__init__(args.mass_area_density, args.thickness, ref_geometry)
        self.poissons_ratio = args.poissons_ratio
        self.D = (args.youngs_modulus * args.thickness) / (1 - args.poissons_ratio ** 2)
        self.B = (args.thickness ** 2) * self.D / 12 
    
    def compute_internal_energy(self, strain: Strain) -> torch.Tensor:
        H__1111, H__1112, H__1122, H__1212, H__1222, H__2222 = self.ref_geometry.elastic_tensor(self.poissons_ratio)
            
        n__1__1 = H__1111 * strain.epsilon_1_1 + 2 * H__1112 * strain.epsilon_1_2 + H__1122 * strain.epsilon_2_2
        n__1__2 = H__1112 * strain.epsilon_1_1 + 2 * H__1212 * strain.epsilon_1_2 + H__1222 * strain.epsilon_2_2
        n__2__1 = n__1__2
        n__2__2 = H__1122 * strain.epsilon_1_1 + 2 * H__1222 * strain.epsilon_1_2 + H__2222 * strain.epsilon_2_2

        m__1__1 = H__1111 * strain.kappa_1_1 + 2 * H__1112 * strain.kappa_1_2 + H__1122 * strain.kappa_2_2
        m__1__2 = H__1112 * strain.kappa_1_1 + 2 * H__1212 * strain.kappa_1_2 + H__1222 * strain.kappa_2_2
        m__2__1 = m__1__2
        m__2__2 = H__1122 * strain.kappa_1_1 + 2 * H__1222 * strain.kappa_1_2 + H__2222 * strain.kappa_2_2
        
        hyperelastic_strain_energy = 0.5 * (self.D * (strain.epsilon_1_1 * n__1__1 + strain.epsilon_1_2 * n__1__2 + strain.epsilon_1_2 * n__2__1 + strain.epsilon_2_2 * n__2__2) + self.B * (strain.kappa_1_1 * m__1__1 + strain.kappa_1_2 * m__1__2 + strain.kappa_1_2 * m__2__1 + strain.kappa_2_2 * m__2__2))
        
        return hyperelastic_strain_energy

class MaterialOrthotropy(NamedTuple):
    d_1: torch.Tensor
    d_2: torch.Tensor
            
class NonLinearMaterial(Material):
    def __init__(self, args, ref_geometry):
        super().__init__(args.mass_area_density, args.thickness, ref_geometry)
        self.a11, self.a12, self.a22, self.G12 = args.a11, args.a12, args.a22, args.G12
        
        self.StVK = args.StVK        
        if not self.StVK:
            self.d = args.d
            self.mu = [args.mu1, args.mu2, args.mu3, args.mu4]
            self.alpha = [args.alpha1, args.alpha2, args.alpha3, args.alpha4]
            
            self.E_11_min, self.E_11_max, self.E_22_min, self.E_22_max, self.E_12_max = args.E_11_min, args.E_11_max, args.E_22_min, args.E_22_max, args.E_12_max
            self.E_12_min = -self.E_12_max
        self.i_debug = args.i_debug        
    
    def compute_eta(self, j: int, x: torch.Tensor) -> torch.Tensor:
        eta = torch.zeros_like(x)
        for i in range(self.d[j]):
            eta += ((self.mu[j][i] / self.alpha[j][i]) * ((x + 1) ** self.alpha[j][i] - 1))
        return eta
    
    def compute_eta_first_derivative(self, j: int, x: torch.Tensor) -> torch.Tensor:
        eta_first_derivative = torch.zeros_like(x)
        for i in range(self.d[j]):
            eta_first_derivative += (self.mu[j][i] * (x + 1) ** (self.alpha[j][i] - 1))
        return eta_first_derivative
    
    def compute_eta_second_derivative(self, j: int, x: torch.Tensor) -> torch.Tensor: 
        eta_second_derivative = torch.zeros_like(x)
        for i in range(self.d[j]):
            eta_second_derivative += (self.mu[j][i] * (self.alpha[j][i] - 1) * (x + 1) ** (self.alpha[j][i] - 2))
        return eta_second_derivative
    
    def strain_cutoff_extrapolation(self, E11, E12, E22, E11_clamped, E12_clamped, E22_clamped, i):        
        E11_valid = torch.logical_and(E11 > self.E_11_min, E11 < self.E_11_max)
        E12_valid = torch.logical_and(E12 > self.E_12_min, E12 < self.E_12_max)
        E22_valid = torch.logical_and(E22 > self.E_22_min, E22 < self.E_22_max)
        
        if not i % self.i_debug and tb.writer:
            tb.writer.add_histogram('param/E11_valid', E11_valid, i)
            tb.writer.add_histogram('param/E12_valid', E12_valid, i)
            tb.writer.add_histogram('param/E22_valid', E22_valid, i)
                
        eta_first_derivative_3_E12_12_clamped = self.compute_eta_first_derivative(3, E12_clamped ** 2)
        eta_first_derivative_2_E22_22_clamped = self.compute_eta_first_derivative(2, E22_clamped ** 2)
        eta_first_derivative_1_E11_22_clamped = self.compute_eta_first_derivative(1, E11_clamped * E22_clamped)
        eta_first_derivative_0_E11_11_clamped = self.compute_eta_first_derivative(0, E11_clamped ** 2)
        eta_second_derivative_1_E11_22_clamped = self.compute_eta_second_derivative(1, E11_clamped * E22_clamped)
        
        extrapolated_hyperelastic_strain_energy = ~E12_valid * (2 * self.G12 * E12_clamped * eta_first_derivative_3_E12_12_clamped * (E12 - E12_clamped) + 0.5 * (2 * self.G12 * eta_first_derivative_3_E12_12_clamped + 4 * self.G12 * E12_clamped ** 2 * self.compute_eta_second_derivative(3, E12_clamped ** 2)) * (E12 - E12_clamped) ** 2)
        extrapolated_hyperelastic_strain_energy += ~E11_valid * ((self.a11 * E11_clamped * eta_first_derivative_0_E11_11_clamped + self.a12 * E22_clamped * eta_first_derivative_1_E11_22_clamped) * (E11 - E11_clamped) + 0.5 * (self.a11 * eta_first_derivative_0_E11_11_clamped + 2 * self.a11 * E11_clamped ** 2 * self.compute_eta_second_derivative(0, E11_clamped ** 2) + self.a12 * E22_clamped ** 2 * eta_second_derivative_1_E11_22_clamped) * (E11 - E11_clamped) ** 2)
        extrapolated_hyperelastic_strain_energy += ~E22_valid * ((self.a22 * E22_clamped * eta_first_derivative_2_E22_22_clamped + self.a12 * E11_clamped * eta_first_derivative_1_E11_22_clamped) * (E22 - E22_clamped) + 0.5 * (self.a22 * eta_first_derivative_2_E22_22_clamped + 2 * self.a22 * E22_clamped ** 2 * self.compute_eta_second_derivative(2, E22_clamped ** 2) + self.a12 * E11_clamped ** 2 * eta_second_derivative_1_E11_22_clamped) * (E22 - E22_clamped) ** 2)
        extrapolated_hyperelastic_strain_energy += (~E11_valid * ~E22_valid) * (self.a12 * eta_first_derivative_1_E11_22_clamped + self.a12 * E11_clamped * E22_clamped * eta_second_derivative_1_E11_22_clamped) * (E11 - E11_clamped) * (E22 - E22_clamped)
        
        return extrapolated_hyperelastic_strain_energy
            
    def compute_internal_energy(self, strain: Strain, material_directions: MaterialOrthotropy, i: int, xi__3: float) -> torch.Tensor:
        # Eq. (22)
        E11_shell_basis = strain.epsilon_1_1 + xi__3 * strain.kappa_1_1
        E12_shell_basis = strain.epsilon_1_2 + xi__3 * strain.kappa_1_2
        E22_shell_basis = strain.epsilon_2_2 + xi__3 * strain.kappa_2_2
        
        g__1__1, g__1__2, g__2__1, g__2__2 = self.ref_geometry.shell_base_vectors(xi__3)        
        E_shell_basis = torch.einsum('ij,ijkl->ijkl', E11_shell_basis, g__1__1) + torch.einsum('ij,ijkl->ijkl', E12_shell_basis, g__1__2) + torch.einsum('ij,ijkl->ijkl', E12_shell_basis, g__2__1) + torch.einsum('ij,ijkl->ijkl', E22_shell_basis, g__2__2)
        
        # In all the subsequent operations, strain components are in the material/orthotropy basis, i.e E_tilde in the supplement Eq. (29)
        E11 = torch.einsum('ijk,ijkl,ijl->ij', material_directions.d_1, E_shell_basis, material_directions.d_1)
        E12 = torch.einsum('ijk,ijkl,ijl->ij', material_directions.d_1, E_shell_basis, material_directions.d_2)
        E22 = torch.einsum('ijk,ijkl,ijl->ij', material_directions.d_2, E_shell_basis, material_directions.d_2)
          
        if self.StVK:
            hyperelastic_strain_energy = self.a11 * 0.5 * E11 ** 2 + self.a12 * E11 * E22 + self.a22 * 0.5 * E22 ** 2 + self.G12 * E12 ** 2                  
        else:                
            E11_clamped = torch.clamp(E11, self.E_11_min, self.E_11_max)
            E12_clamped = torch.clamp(E12, self.E_12_min, self.E_12_max)
            E22_clamped = torch.clamp(E22, self.E_22_min, self.E_22_max)
            
            hyperelastic_strain_energy = self.a11 * 0.5 * self.compute_eta(0, E11_clamped ** 2) + self.a12 * self.compute_eta(1, E11_clamped * E22_clamped) + self.a22 * 0.5 * self.compute_eta(2, E22_clamped ** 2) + self.G12 * self.compute_eta(3, E12_clamped ** 2)            
            hyperelastic_strain_energy += self.strain_cutoff_extrapolation(E11, E12, E22, E11_clamped, E12_clamped, E22_clamped, i)          
            
        if not i % self.i_debug and tb.writer:
            tb.writer.add_histogram('param/E11', E11, i)
            tb.writer.add_histogram('param/E12', E12, i)
            tb.writer.add_histogram('param/E22', E22, i)
        
        return hyperelastic_strain_energy