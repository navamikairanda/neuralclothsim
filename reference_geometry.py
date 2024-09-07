import torch
from torch.nn.functional import normalize

from diff_operators import jacobian
from plot_helper import get_plot_grid_tensor
from reference_midsurface import ReferenceMidSurface

class ReferenceGeometry(): 
    def __init__(self, curvilinear_coords: torch.Tensor, n_spatial_samples: int, temporal_sidelen: int, reference_midsurface: ReferenceMidSurface, tb_writer, debug=False):
        self.curvilinear_coords, self.n_spatial_samples, self.temporal_sidelen = curvilinear_coords, n_spatial_samples, temporal_sidelen
        self.midsurface_positions = reference_midsurface.midsurface(self.curvilinear_coords)
        self.base_vectors()
        self.metric_tensor()
        self.curvature_tensor()
        self.christoffel_symbol()
        self.coord_transform()   
        self.curvilinear_coords = self.curvilinear_coords.repeat(1, self.temporal_sidelen, 1)
        self.midsurface_positions = self.midsurface_positions.repeat(1, self.temporal_sidelen, 1)
        self.a_1 = self.a_1.repeat(1, self.temporal_sidelen, 1)
        self.a_2 = self.a_2.repeat(1, self.temporal_sidelen, 1)
        self.a_3 = self.a_3.repeat(1, self.temporal_sidelen, 1)
        self.a__1 = self.a__1.repeat(1, self.temporal_sidelen, 1)
        self.a__2 = self.a__2.repeat(1, self.temporal_sidelen, 1)
        if debug:       
            tb_writer.add_figure('metric_tensor', get_plot_grid_tensor(self.a_1_1[0], self.a_1_2[0],self.a_1_2[0], self.a_2_2[0]))
            tb_writer.add_figure('curvature_tensor', get_plot_grid_tensor(self.b_1_1[0,:n_spatial_samples], self.b_1_2[0,:n_spatial_samples],self.b_2_1[0,:n_spatial_samples], self.b_2_2[0,:n_spatial_samples]))            
            
    def base_vectors(self):
        base_vectors = jacobian(self.midsurface_positions, self.curvilinear_coords)[0]
        self.a_1 = base_vectors[...,0]
        self.a_2 = base_vectors[...,1]
        self.a_3 = normalize(torch.linalg.cross(self.a_1, self.a_2), dim=2)
              
    def metric_tensor(self):
        self.a_1_1 = torch.einsum('ijk,ijk->ij', self.a_1, self.a_1)
        self.a_1_2 = torch.einsum('ijk,ijk->ij', self.a_1, self.a_2)
        self.a_2_2 = torch.einsum('ijk,ijk->ij', self.a_2, self.a_2)

        self.a = self.a_1_1 * self.a_2_2 - (self.a_1_2 ** 2)
        self.a__1__1 = self.a_2_2 / self.a
        self.a__2__2 = self.a_1_1 / self.a
        self.a__1__2 = -1 * self.a_1_2 / self.a
        self.a__2__1 = self.a__1__2
        self.a__1 = torch.einsum('ij,ijk->ijk', self.a__1__1, self.a_1) + torch.einsum('ij,ijk->ijk', self.a__1__2, self.a_2)
        self.a__2 = torch.einsum('ij,ijk->ijk', self.a__1__2, self.a_1) + torch.einsum('ij,ijk->ijk', self.a__2__2, self.a_2)
        
        self.a__1__1 = self.a__1__1.repeat(1, self.temporal_sidelen)
        self.a__2__2 = self.a__2__2.repeat(1, self.temporal_sidelen)
        self.a__1__2 = self.a__1__2.repeat(1, self.temporal_sidelen)
        self.a__2__1 = self.a__2__1.repeat(1, self.temporal_sidelen)
        with torch.no_grad():
            self.a = self.a.repeat(1, self.temporal_sidelen)
    
    def curvature_tensor(self):        
        self.a_3pd = jacobian(self.a_3, self.curvilinear_coords)[0]
        self.b_1_1 = -1 * torch.einsum('ijk,ijk->ij', self.a_1, self.a_3pd[...,0])
        self.b_1_2 = -1 * torch.einsum('ijk,ijk->ij', self.a_1, self.a_3pd[...,1])
        self.b_2_2 = -1 * torch.einsum('ijk,ijk->ij', self.a_2, self.a_3pd[...,1])
        self.b_2_1 = self.b_1_2

        self.b_1_1 = self.b_1_1.repeat(1, self.temporal_sidelen)
        self.b_1_2 = self.b_1_2.repeat(1, self.temporal_sidelen)
        self.b_2_2 = self.b_2_2.repeat(1, self.temporal_sidelen)
        self.b_2_1 = self.b_2_1.repeat(1, self.temporal_sidelen)

        self.b_1__1 = self.b_1_1 * self.a__1__1 + self.b_1_2 * self.a__2__1
        self.b_1__2 = self.b_1_1 * self.a__1__2 + self.b_1_2 * self.a__2__2
        self.b_2__1 = self.b_2_1 * self.a__1__1 + self.b_2_2 * self.a__2__1
        self.b_2__2 = self.b_2_1 * self.a__1__2 + self.b_2_2 * self.a__2__2

    def coord_transform(self):
        with torch.no_grad():
            #contravariant_coord_2_cartesian = torch.stack([self.a_1, self.a_2, self.a_3], dim=3)
            #self.cartesian_coord_2_contravariant = torch.linalg.inv(contravariant_coord_2_cartesian)            
            covariant_coord_2_cartesian = torch.stack([self.a__1, self.a__2, self.a_3], dim=3)
            self.cartesian_coord_2_covariant = torch.linalg.inv(covariant_coord_2_cartesian)        
        self.cartesian_coord_2_covariant = self.cartesian_coord_2_covariant.repeat(1,self.temporal_sidelen,1,1)
    
    def christoffel_symbol(self):
        a_1pd = jacobian(self.a_1, self.curvilinear_coords)[0]
        a_2pd = jacobian(self.a_2, self.curvilinear_coords)[0]

        with torch.no_grad():
            self.gamma_11__1 = torch.einsum('ijk,ijk->ij', a_1pd[...,0], self.a__1)
            self.gamma_12__1 = torch.einsum('ijk,ijk->ij', a_1pd[...,1], self.a__1)
            self.gamma_22__1 = torch.einsum('ijk,ijk->ij', a_2pd[...,1], self.a__1)

            self.gamma_11__2 = torch.einsum('ijk,ijk->ij', a_1pd[...,0], self.a__2)
            self.gamma_12__2 = torch.einsum('ijk,ijk->ij', a_1pd[...,1], self.a__2)
            self.gamma_22__2 = torch.einsum('ijk,ijk->ij', a_2pd[...,1], self.a__2)
            self.gamma_21__1 = self.gamma_12__1
            self.gamma_21__2 = self.gamma_12__2
        self.gamma_11__1 = self.gamma_11__1.repeat(1,self.temporal_sidelen)
        self.gamma_12__1 = self.gamma_12__1.repeat(1,self.temporal_sidelen)
        self.gamma_22__1 = self.gamma_22__1.repeat(1,self.temporal_sidelen)
        self.gamma_11__2 = self.gamma_11__2.repeat(1,self.temporal_sidelen)
        self.gamma_12__2 = self.gamma_12__2.repeat(1,self.temporal_sidelen)
        self.gamma_22__2 = self.gamma_22__2.repeat(1,self.temporal_sidelen)
        self.gamma_21__1 = self.gamma_21__1.repeat(1,self.temporal_sidelen)
        self.gamma_21__2 = self.gamma_21__2.repeat(1,self.temporal_sidelen)

    @torch.no_grad()
    def elastic_tensor(self, poissons_ratio: float):
        H__1111 = poissons_ratio * self.a__1__1 * self.a__1__1 + 0.5 * (1 - poissons_ratio) * (self.a__1__1 * self.a__1__1 + self.a__1__1 * self.a__1__1)
        H__1112 = poissons_ratio * self.a__1__1 * self.a__1__2 + 0.5 * (1 - poissons_ratio) * (self.a__1__1 * self.a__1__2 + self.a__1__2 * self.a__1__1)
        H__1122 = poissons_ratio * self.a__1__1 * self.a__2__2 + 0.5 * (1 - poissons_ratio) * (self.a__1__2 * self.a__1__2 + self.a__1__2 * self.a__1__2)
        H__1212 = poissons_ratio * self.a__1__2 * self.a__1__2 + 0.5 * (1 - poissons_ratio) * (self.a__1__1 * self.a__2__2 + self.a__1__2 * self.a__2__1)
        H__1222 = poissons_ratio * self.a__1__2 * self.a__2__2 + 0.5 * (1 - poissons_ratio) * (self.a__1__2 * self.a__2__2 + self.a__1__2 * self.a__2__2)
        H__2222 = poissons_ratio * self.a__2__2 * self.a__2__2 + 0.5 * (1 - poissons_ratio) * (self.a__2__2 * self.a__2__2 + self.a__2__2 * self.a__2__2)
        return H__1111, H__1112, H__1122, H__1212, H__1222, H__2222
        
    def shell_base_vectors(self, xi__3: float):
        g_1_1 = self.a_1_1.repeat(1, self.temporal_sidelen) - 2 * xi__3 * self.b_1_1
        g_1_2 = self.a_1_2.repeat(1, self.temporal_sidelen) - 2 * xi__3 * self.b_1_2
        g_2_2 = self.a_2_2.repeat(1, self.temporal_sidelen) - 2 * xi__3 * self.b_2_2
        
        g_1 = self.a_1 + xi__3 * self.a_3pd[...,0].repeat(1, self.temporal_sidelen, 1)
        g_2 = self.a_2 + xi__3 * self.a_3pd[...,1].repeat(1, self.temporal_sidelen, 1)
        g_covariant_matrix = torch.stack([torch.stack([g_1_1, g_1_2], dim=2), torch.stack([g_1_2, g_2_2], dim=2)], dim=2) 
        g_contravariant_matrix = torch.linalg.inv(g_covariant_matrix)
        g__1 = torch.einsum('ij,ijk->ijk', g_contravariant_matrix[...,0,0], g_1) + torch.einsum('ij,ijk->ijk', g_contravariant_matrix[...,0,1], g_2)
        g__2 = torch.einsum('ij,ijk->ijk', g_contravariant_matrix[...,1,0], g_1) + torch.einsum('ij,ijk->ijk', g_contravariant_matrix[...,1,1], g_2)

        g__1__1 = torch.einsum('ijk,ijl->ijkl', g__1, g__1)
        g__1__2 = torch.einsum('ijk,ijl->ijkl', g__1, g__2)
        g__2__1 = torch.einsum('ijk,ijl->ijkl', g__2, g__1)
        g__2__2 = torch.einsum('ijk,ijl->ijkl', g__2, g__2)
        return g__1__1, g__1__2, g__2__1, g__2__2    