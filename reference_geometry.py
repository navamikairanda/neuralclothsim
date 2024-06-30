import torch
from torch.nn.functional import normalize

from diff_operators import jacobian
from helper import grads2img, get_plot_grid_tensor, get_plot_single_tensor

class ReferenceGeometry(): 
    def __init__(self, curvilinear_coords, spatial_sidelen, temporal_sidelen, reference_midsurface, tb_writer, debug_ref_geometry=False):
        self.curvilinear_coords, self.spatial_sidelen, self.temporal_sidelen = curvilinear_coords, spatial_sidelen, temporal_sidelen
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
        if debug_ref_geometry:
            a = torch.stack([self.a_1, self.a_2], dim=3)[0].view(spatial_sidelen, spatial_sidelen, 3, 2)
            tb_writer.add_image('a_x', grads2img(a[...,0,:]))
            tb_writer.add_image('a_y', grads2img(a[...,1,:]))
            tb_writer.add_image('a_z', grads2img(a[...,2,:]))           
            tb_writer.add_figure('metric_tensor', get_plot_grid_tensor(self.a_1_1[0], self.a_1_2[0],self.a_1_2[0], self.a_2_2[0], spatial_sidelen))
            tb_writer.add_figure('curvature_tensor', get_plot_grid_tensor(self.b_1_1[0,:curvilinear_coords.shape[1]], self.b_1_2[0,:curvilinear_coords.shape[1]],self.b_2_1[0,:curvilinear_coords.shape[1]], self.b_2_2[0,:curvilinear_coords.shape[1]], spatial_sidelen))            
            #tb_writer.add_figure('curvilinear_tensor', get_plot_single_tensor(self.midsurface_positions[0,:curvilinear_coords.shape[1],0], spatial_sidelen))
            
    def base_vectors(self):
        base_vectors = jacobian(self.midsurface_positions, self.curvilinear_coords)[0]
        self.a_1 = base_vectors[...,0]
        self.a_2 = base_vectors[...,1]
        self.a_3 = normalize(torch.linalg.cross(self.a_1, self.a_2), dim=2)
              
    def metric_tensor(self):    
        '''
        a_1_1 = torch.einsum('ijk,ijk->ij', self.a_1, self.a_1)
        a_1_2 = torch.einsum('ijk,ijk->ij', self.a_1, self.a_2)
        a_2_2 = torch.einsum('ijk,ijk->ij', self.a_2, self.a_2)

        self.a = a_1_1 * a_2_2 - (a_1_2 ** 2)
        self.a__1__1 = a_2_2 / self.a
        self.a__2__2 = a_1_1 / self.a
        self.a__1__2 = -1 * a_1_2 / self.a
        self.a__2__1 = self.a__1__2
        '''
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
            contravariant_coord_2_cartesian = torch.stack([self.a_1, self.a_2, self.a_3], dim=3)
            self.cartesian_coord_2_contravariant = torch.linalg.inv(contravariant_coord_2_cartesian)
            
            covariant_coord_2_cartesian = torch.stack([self.a__1, self.a__2, self.a_3], dim=3)
            self.cartesian_coord_2_covariant = torch.linalg.inv(covariant_coord_2_cartesian)
        
        self.cartesian_coord_2_contravariant = self.cartesian_coord_2_contravariant.repeat(1,self.temporal_sidelen,1,1)
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
