import torch

class Material():
    def __init__(self, mass_area_density: float, thickness: float):
        self.thickness = thickness
        self.mass_area_density = mass_area_density
        
class LinearMaterial(Material):
    def __init__(self, args):
        super().__init__(args.mass_area_density, args.thickness)
        self.poissons_ratio = args.poissons_ratio
        self.D = (args.youngs_modulus * args.thickness) / (1 - args.poissons_ratio ** 2)
        self.B = (args.thickness ** 2) * self.D / 12 
        
class NonLinearMaterial(Material):
    def __init__(self, args):
        super().__init__(args.mass_area_density, args.thickness)
        self.a11, self.a12, self.a22, self.G12 = args.a11, args.a12, args.a22, args.G12
        
        self.StVK = args.StVK        
        if not self.StVK:
            self.d = args.d
            self.mu = [args.mu1, args.mu2, args.mu3, args.mu4]
            self.alpha = [args.alpha1, args.alpha2, args.alpha3, args.alpha4]
            
            self.E_11_min, self.E_11_max, self.E_22_min, self.E_22_max, self.E_12_max = args.E_11_min, args.E_11_max, args.E_22_min, args.E_22_max, args.E_12_max
            self.E_12_min = -self.E_12_max        
    
    def compute_eta(self, j: int, x: torch.Tensor) -> torch.Tensor:
        eta = 0
        for i in range(self.d[j]):
            eta += ((self.mu[j][i] / self.alpha[j][i]) * ((x + 1) ** self.alpha[j][i] - 1))
        return eta
    
    def compute_eta_first_derivative(self, j: int, x: torch.Tensor) -> torch.Tensor:
        eta_first_derivative = 0
        for i in range(self.d[j]):
            eta_first_derivative += (self.mu[j][i] * (x + 1) ** (self.alpha[j][i] - 1))
        return eta_first_derivative
    
    def compute_eta_second_derivative(self, j: int, x: torch.Tensor) -> torch.Tensor: 
        eta_second_derivative = 0
        for i in range(self.d[j]):
            eta_second_derivative += (self.mu[j][i] * (self.alpha[j][i] - 1) * (x + 1) ** (self.alpha[j][i] - 2))
        return eta_second_derivative