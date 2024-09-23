import torch
import os
from tqdm import trange
from shutil import copyfile

from torch.utils.data import DataLoader
import utils.tb as tb
from material import LinearMaterial, NonLinearMaterial
from sampler import GridSampler, MeshSampler, CurvilinearSpace
from reference_geometry import ReferenceGeometry
from modules import Siren
from energy import Energy
from utils.logger import get_logger
from utils.config_parser import get_config_parser, device
from reference_midsurface import ReferenceMidSurface
from boundary import Boundary
from utils.file_io import save_meshes
#torch.manual_seed(2) #Set seed for reproducible results

import torch
import torch.nn as nn
import numpy as np
from boundary import Boundary

class SineLayer(nn.Module):      
    def __init__(self, in_features: int, out_features: int, bias=True, is_first=False, omega_0=30.):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
   
class Siren(nn.Module):
    def __init__(self, boundary: Boundary, in_features=3, hidden_features=512, hidden_layers=5, out_features=3, outermost_linear=True, first_omega_0=30., hidden_omega_0=30.):
        super().__init__()
        self.boundary = boundary
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))    
        self.net = nn.Sequential(*self.net)
    
    def forward(self, curvilinear_coords: torch.Tensor) -> torch.Tensor:                

        normalized_coords = self.boundary.periodic_condition_and_normalization(curvilinear_coords)        
        deformations = self.net(normalized_coords)
        deformations = self.boundary.dirichlet_condition(deformations, curvilinear_coords)

        return deformations 

import math
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.config_parser import device

# Parts of the code (sample_points_from_meshes) borrowed from Meta Platforms, Inc.
from typing import Tuple, Union
from pytorch3d.structures import Meshes
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.packed_to_padded import packed_to_padded
from typing import NamedTuple

def sample_points_from_meshes(
    meshes,
    num_samples: int = 10000
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    Convert a batch of meshes to a batch of pointclouds by uniformly sampling
    points on the surface of the mesh with probability proportional to the
    face area.

    Args:
        meshes: A Meshes object with a batch of N meshes.
        num_samples: Integer giving the number of point samples per mesh.

    Returns:
        3-element tuple containing

        - **samples_xyz**: FloatTensor of shape (N, num_samples, 3) giving the
          coordinates of sampled points for each mesh in the batch. For empty
          meshes the corresponding row in the samples array will be filled with 0.
        - **samples_uv**: FloatTensor of shape (N, num_samples, 2) giving the uv coodinates
          for each sampled point. For empty meshes the corresponding row in the array will
          be filled with 0.
    """
    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")

    faces = meshes.faces_packed()
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
    num_meshes = len(meshes)
    num_valid_meshes = torch.sum(meshes.valid)  # Non empty meshes.

    # Initialize samples tensor with fill value 0 for empty meshes.
    samples_xyz = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

    # Only compute samples for non empty meshes
    with torch.no_grad():
        areas, _ = mesh_face_areas_normals(verts, faces)  # Face areas can be zero.
        max_faces = meshes.num_faces_per_mesh().max().item()
        areas_padded = packed_to_padded(
            areas, mesh_to_face[meshes.valid], max_faces
        )  # (N, F)

        # TODO (gkioxari) Confirm multinomial bug is not present with real data.
        sample_face_idxs = areas_padded.multinomial(
            num_samples, replacement=True
        )  # (N, num_samples)
        sample_face_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes, 1)

    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Randomly generate barycentric coords.
    w0, w1, w2 = _rand_barycentric_coords(
        num_valid_meshes, num_samples, verts.dtype, verts.device
    )

    # Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_face_idxs]  # (N, num_samples, 3)
    b = v1[sample_face_idxs]
    c = v2[sample_face_idxs]
    samples_xyz[meshes.valid] = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c
    
    face_uvs = meshes.textures.verts_uvs_padded()[0][meshes.textures.faces_uvs_padded()[0]]
    uv0, uv1, uv2 = face_uvs[:, 0], face_uvs[:, 1], face_uvs[:, 2]
    
    # Use the barycentric coords to get the uv coodinate for the sampled point.
    a_uv = uv0[sample_face_idxs]  # (N, num_samples, 3)
    b_uv = uv1[sample_face_idxs]
    c_uv = uv2[sample_face_idxs]
    samples_uv = torch.zeros((num_meshes, num_samples, 2), device=meshes.device)
    samples_uv[meshes.valid] = w0[:, :, None] * a_uv + w1[:, :, None] * b_uv + w2[:, :, None] * c_uv
    
    return samples_xyz, samples_uv

def _rand_barycentric_coords(
    size1, size2, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper function to generate random barycentric coordinates which are uniformly
    distributed over a triangle.

    Args:
        size1, size2: The number of coordinates generated will be size1*size2.
                      Output tensors will each be of shape (size1, size2).
        dtype: Datatype to generate.
        device: A torch.device object on which the outputs will be allocated.

    Returns:
        w0, w1, w2: Tensors of shape (size1, size2) giving random barycentric
            coordinates
    """
    uv = torch.rand(2, size1, size2, dtype=dtype, device=device)
    u, v = uv[0], uv[1]
    u_sqrt = u.sqrt()
    w0 = 1.0 - u_sqrt
    w1 = u_sqrt * (1.0 - v)
    w2 = u_sqrt * v
    return w0, w1, w2

def get_mgrid(sidelen: Union[Tuple[int], Tuple[int, int]], stratified=False, dim=2):
    # Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    if dim == 1:
        sidelen_x = sidelen[0] if stratified else sidelen[0] - 1
        grid_coords = np.stack(np.mgrid[:sidelen[0]], axis=-1)[None, ..., None].astype(np.float32)
        grid_coords[..., 0] = grid_coords[..., 0] / sidelen_x
    elif dim == 2:
        grid_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        if stratified: 
            sidelen_x, sidelen_y = sidelen[0], sidelen[1]
        else: 
            sidelen_x, sidelen_y = sidelen[0] - 1, sidelen[1] - 1
        grid_coords[0, :, :, 0] = grid_coords[0, :, :, 0] / sidelen_x
        grid_coords[0, :, :, 1] = grid_coords[0, :, :, 1] / sidelen_y
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)
    grid_coords = torch.Tensor(grid_coords).to(device).view(-1, dim)
    return grid_coords

class CurvilinearSpace(NamedTuple):
    xi__1_max: float
    xi__2_max: float
    
class Sampler(Dataset):
    def __init__(self, n_spatial_samples: int):
        self.n_spatial_samples = n_spatial_samples
    
    def __len__(self):
        return 1
                           
class GridSampler(Sampler):
    def __init__(self, n_spatial_samples: int, curvilinear_space: CurvilinearSpace):
        super().__init__(n_spatial_samples)
                
        self.curvilinear_space = curvilinear_space
        self.spatial_sidelen = math.isqrt(n_spatial_samples)
        
        self.cell_curvilinear_coords = get_mgrid((self.spatial_sidelen, self.spatial_sidelen), stratified=True, dim=2)            

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
         
        curvilinear_coords = self.cell_curvilinear_coords.clone()        
        
        t_rand_spatial = torch.rand([self.n_spatial_samples, 2], device=device) / self.spatial_sidelen        
        curvilinear_coords += t_rand_spatial
        
        curvilinear_coords[...,0] *= self.curvilinear_space.xi__1_max
        curvilinear_coords[...,1] *= self.curvilinear_space.xi__2_max
        curvilinear_coords.requires_grad_(True)
      
        return curvilinear_coords

class MeshSampler(Sampler):
    def __init__(self, n_spatial_samples: int, reference_mesh: Meshes):
        super().__init__(n_spatial_samples)
        self.reference_mesh = reference_mesh
            
    def __getitem__(self, idx):    
        if idx > 0: raise IndexError

        curvilinear_coords = sample_points_from_meshes(self.reference_mesh, self.n_spatial_samples)[1][0]            
        curvilinear_coords.requires_grad_(True)
                    
        return curvilinear_coords

import torch
from torch.autograd import grad
from typing import Tuple

def jacobian(y: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    ''' jacobian of y wrt x '''
    meta_batch_size, num_observations = y.shape[:2]
    jac = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[...,i].view(-1, 1)
        jac[:, :, i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True, allow_unused=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1
    return jac, status
import os
import math
import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
from pytorch3d.io import save_obj, load_obj

import utils.tb as tb
from utils.config_parser import device
from modules import GELUReference
from sampler import get_mgrid, sample_points_from_meshes, CurvilinearSpace
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import TexturesUV

def generate_mesh_topology(spatial_sidelen):
    rows = cols = spatial_sidelen
    last_face_index = cols * (rows - 1)
    
    first_face_bl = [0, cols, 1]  
    first_face_tr = [cols + 1, 1, cols]  
    all_faces = []
    for first_face in [first_face_bl, first_face_tr]:
        last_face = [i + last_face_index - 1 for i in first_face]
        faces = np.linspace(first_face, last_face, last_face_index)
        faces = np.reshape(faces, (rows - 1, cols, 3))
        faces = np.delete(faces, cols - 1, 1)
        faces = np.reshape(faces, (-1, 3))   
        all_faces.append(faces)
    return np.concatenate(all_faces, axis=0)

class ReferenceMidSurface():
    def __init__(self, args, curvilinear_space: CurvilinearSpace):
        self.reference_geometry_name = args.reference_geometry_name
        self.boundary_curvilinear_coords = None
        if self.reference_geometry_name == 'mesh':
            vertices, faces, aux = load_obj(args.reference_geometry_source, load_textures=False, device=device)
            texture = TexturesUV(maps=torch.empty(1, 1, 1, 1, device=device), faces_uvs=[faces.textures_idx], verts_uvs=[aux.verts_uvs])
            self.template_mesh = Meshes(verts=[vertices], faces=[faces.verts_idx], textures=texture).to(device)
            if args.boundary_condition_name == 'mesh_vertices':
                self.boundary_curvilinear_coords = self.template_mesh.textures.verts_uvs_padded()[0][args.reference_boundary_vertices]
            self.fit_reference_mlp(args.reference_mlp_lrate, args.reference_mlp_n_iterations)
            reference_mlp_verts_pred = self(self.template_mesh.textures.verts_uvs_padded())
            self.template_mesh = self.template_mesh.update_padded(reference_mlp_verts_pred)            
        else:
            # for analytical surface, use equal number of samples along each curvilinear coordinate
            args.train_n_spatial_samples, args.test_n_spatial_samples = math.isqrt(args.train_n_spatial_samples) ** 2, math.isqrt(args.test_n_spatial_samples) ** 2
            test_spatial_sidelen = math.isqrt(args.test_n_spatial_samples)
            curvilinear_coords = get_mgrid((test_spatial_sidelen, test_spatial_sidelen), stratified=False, dim=2)[None]
            curvilinear_coords[...,0] *= curvilinear_space.xi__1_max
            curvilinear_coords[...,1] *= curvilinear_space.xi__2_max
            vertices = self(curvilinear_coords)[0]
            faces = torch.tensor(generate_mesh_topology(test_spatial_sidelen), device=device)
            texture = TexturesUV(maps=torch.empty(1, 1, 1, 1, device=device), faces_uvs=[faces], verts_uvs=curvilinear_coords)
            self.template_mesh = Meshes(verts=[vertices], faces=[faces], textures=texture).to(device)
        if tb.writer:
            tb.writer.add_mesh('reference_state', self.template_mesh.verts_padded(), faces=self.template_mesh.textures.faces_uvs_padded())
        save_obj(os.path.join(args.logging_dir, args.expt_name, 'reference_state.obj'), self.template_mesh.verts_packed(), self.template_mesh.textures.faces_uvs_padded()[0], verts_uvs=self.template_mesh.textures.verts_uvs_padded()[0], faces_uvs=self.template_mesh.textures.faces_uvs_padded()[0])
        
    def fit_reference_mlp(self, reference_mlp_lrate: float, reference_mlp_n_iterations: int):
        self.reference_mlp = GELUReference(in_features=2, hidden_features=512, out_features=3, hidden_layers=5).to(device)
        reference_optimizer = torch.optim.Adam(lr=reference_mlp_lrate, params=self.reference_mlp.parameters())
        loss_fn = nn.L1Loss()        
        for i in trange(reference_mlp_n_iterations):
            reference_optimizer.zero_grad()
            with torch.no_grad(): 
                verts, uvs = sample_points_from_meshes(self.template_mesh, 400)           
            loss = loss_fn(self.reference_mlp(uvs), verts)
            loss.backward()
            reference_optimizer.step()
            if tb.writer:
                tb.writer.add_scalar('loss/reference_fitting_loss', loss.detach().item(), i)        
        
    def __call__(self, curvilinear_coords: torch.Tensor) -> torch.Tensor:
        xi__1 = curvilinear_coords[...,0] 
        xi__2 = curvilinear_coords[...,1]
        match self.reference_geometry_name:
            case 'rectangle_xy': #vertical
                midsurface_positions = torch.stack([xi__1, xi__2, 0.* (xi__1**2 - xi__2**2)], dim=2)
            case 'rectangle_xz': #horizontal
                midsurface_positions = torch.stack([xi__1, 0.* (xi__1**2 - xi__2**2), xi__2], dim=2)
            case 'cylinder':
                R = 0.25
                midsurface_positions = torch.stack([R * torch.cos(xi__1), xi__2, R * torch.sin(xi__1)], dim=2)
            case 'cone':
                R_top, R_bottom, L = 0.2, 1.5, 1
                R = xi__2 * (R_top - R_bottom) / L + R_bottom
                midsurface_positions = torch.stack([R * torch.cos(xi__1), xi__2, R * torch.sin(xi__1)], dim=2)
            case 'mesh':  
                midsurface_positions = self.reference_mlp(curvilinear_coords)
            case _: 
                raise ValueError(f'Unknown reference_geometry_name {self.reference_geometry_name}')
        return midsurface_positions
    
import torch
from torch.nn.functional import normalize

import utils.tb as tb
from utils.diff_operators import jacobian
from utils.plot import get_plot_grid_tensor
from reference_midsurface import ReferenceMidSurface

class ReferenceGeometry(): 
    def __init__(self, reference_midsurface: ReferenceMidSurface, n_spatial_samples: int):
        self.n_spatial_samples = n_spatial_samples 
        self.reference_midsurface = reference_midsurface

    def __call__(self, curvilinear_coords: torch.Tensor, debug: bool):
        self.curvilinear_coords = curvilinear_coords
        self.midsurface_positions = self.reference_midsurface(self.curvilinear_coords)
        self.base_vectors()
        self.metric_tensor()
        self.curvature_tensor()
        self.christoffel_symbol()
        self.coord_transform()
        if debug and tb.writer and self.reference_midsurface.reference_geometry_name != 'mesh':
            tb.writer.add_figure('metric_tensor', get_plot_grid_tensor(self.a_1_1[0], self.a_1_2[0],self.a_1_2[0], self.a_2_2[0]))
            tb.writer.add_figure('curvature_tensor', get_plot_grid_tensor(self.b_1_1[0,:self.n_spatial_samples], self.b_1_2[0,:self.n_spatial_samples],self.b_2_1[0,:self.n_spatial_samples], self.b_2_2[0,:self.n_spatial_samples]))
            
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
    
    def curvature_tensor(self):        
        self.a_3pd = jacobian(self.a_3, self.curvilinear_coords)[0]
        self.b_1_1 = -1 * torch.einsum('ijk,ijk->ij', self.a_1, self.a_3pd[...,0])
        self.b_1_2 = -1 * torch.einsum('ijk,ijk->ij', self.a_1, self.a_3pd[...,1])
        self.b_2_2 = -1 * torch.einsum('ijk,ijk->ij', self.a_2, self.a_3pd[...,1])
        self.b_2_1 = self.b_1_2

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
        g_1_1 = self.a_1_1 - 2 * xi__3 * self.b_1_1
        g_1_2 = self.a_1_2 - 2 * xi__3 * self.b_1_2
        g_2_2 = self.a_2_2 - 2 * xi__3 * self.b_2_2
        
        g_1 = self.a_1 + xi__3 * self.a_3pd[...,0]
        g_2 = self.a_2 + xi__3 * self.a_3pd[...,1]
        g_covariant_matrix = torch.stack([torch.stack([g_1_1, g_1_2], dim=2), torch.stack([g_1_2, g_2_2], dim=2)], dim=2) 
        g_contravariant_matrix = torch.linalg.inv(g_covariant_matrix)
        g__1 = torch.einsum('ij,ijk->ijk', g_contravariant_matrix[...,0,0], g_1) + torch.einsum('ij,ijk->ijk', g_contravariant_matrix[...,0,1], g_2)
        g__2 = torch.einsum('ij,ijk->ijk', g_contravariant_matrix[...,1,0], g_1) + torch.einsum('ij,ijk->ijk', g_contravariant_matrix[...,1,1], g_2)

        g__1__1 = torch.einsum('ijk,ijl->ijkl', g__1, g__1)
        g__1__2 = torch.einsum('ijk,ijl->ijkl', g__1, g__2)
        g__2__1 = torch.einsum('ijk,ijl->ijkl', g__2, g__1)
        g__2__2 = torch.einsum('ijk,ijl->ijkl', g__2, g__2)
        return g__1__1, g__1__2, g__2__1, g__2__2    

import torch
from typing import NamedTuple
import utils.tb as tb
from utils.diff_operators import jacobian
from reference_geometry import ReferenceGeometry
from utils.plot import get_plot_grid_tensor

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

def compute_strain(deformations: torch.Tensor, ref_geometry: ReferenceGeometry, i: int, i_debug: int, nonlinear_strain=True): 
    
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
    
    if not i % i_debug and tb.writer and ref_geometry.reference_midsurface.reference_geometry_name != 'mesh':
        tb.writer.add_figure(f'membrane_strain', get_plot_grid_tensor(epsilon_1_1[0,-ref_geometry.n_spatial_samples:], epsilon_1_2[0,-ref_geometry.n_spatial_samples:], epsilon_1_2[0,-ref_geometry.n_spatial_samples:], epsilon_2_2[0,-ref_geometry.n_spatial_samples:]), i)
        tb.writer.add_figure(f'bending_strain', get_plot_grid_tensor(kappa_1_1[0,-ref_geometry.n_spatial_samples:], kappa_1_2[0,-ref_geometry.n_spatial_samples:], kappa_1_2[0,-ref_geometry.n_spatial_samples:], kappa_2_2[0,-ref_geometry.n_spatial_samples:]), i)
    
    return Strain(epsilon_1_1, epsilon_1_2, epsilon_2_2, kappa_1_1, kappa_1_2, kappa_2_2), normal_difference

import torch
from typing import NamedTuple
import utils.tb as tb
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
    def __init__(self, args, ref_geometry: ReferenceGeometry):
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

import math
import torch
from sampler import CurvilinearSpace

class Boundary:
    def __init__(self, reference_geometry_name: str, boundary_condition_name: str, curvilinear_space: CurvilinearSpace, boundary_curvilinear_coords: torch.Tensor = None):
        self.reference_geometry_name = reference_geometry_name
        self.boundary_condition_name = boundary_condition_name
        self.curvilinear_space = curvilinear_space
        self.boundary_curvilinear_coords = boundary_curvilinear_coords
        self.boundary_support = 0.01

    def periodic_condition_and_normalization(self, curvilinear_coords: torch.Tensor) -> torch.Tensor:
        if self.reference_geometry_name in ['cylinder', 'cone']:
            normalized_coords = torch.cat([(torch.cos(curvilinear_coords[...,0:1]) + 1)/2, (torch.sin(curvilinear_coords[...,0:1]) + 1)/2, curvilinear_coords[...,1:2]/self.curvilinear_space.xi__2_max], dim=2)
        else:
            normalized_coords = torch.cat([curvilinear_coords[...,0:1]/self.curvilinear_space.xi__1_max, curvilinear_coords[...,1:2]/self.curvilinear_space.xi__2_max], dim=2)
        return normalized_coords
    
    def dirichlet_condition(self, deformations: torch.Tensor, curvilinear_coords: torch.Tensor) -> torch.Tensor:        
        match self.boundary_condition_name:
            case 'top_left_fixed':
                top_left_corner = torch.exp(-(curvilinear_coords[...,0:1] ** 2 + (curvilinear_coords[...,1:2] - self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)
                deformations = deformations * (1 - top_left_corner)
            case 'top_left_top_right_moved':
                top_left_corner = torch.exp(-(curvilinear_coords[...,0:1] ** 2 + (curvilinear_coords[...,1:2] - self.curvilinear_space.xi__1_max) ** 2)/self.boundary_support)
                top_right_corner = torch.exp(-((curvilinear_coords[...,0:1] - self.curvilinear_space.xi__1_max) ** 2 + (curvilinear_coords[...,1:2] - self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)                
                temporal_motion = 0.2 * torch.ones_like(curvilinear_coords[...,0:1])               
                corner_displacement = torch.cat([temporal_motion, torch.zeros_like(temporal_motion), torch.zeros_like(temporal_motion)], dim=2)                    
                deformations = deformations * (1 - top_left_corner) * (1 - top_right_corner) + corner_displacement * top_left_corner - corner_displacement * top_right_corner                                
            case 'adjacent_edges_fixed':
                left_edge = torch.exp(-(curvilinear_coords[...,0:1] ** 2)/self.boundary_support)
                right_edge = torch.exp(-(curvilinear_coords[...,1:2] ** 2)/self.boundary_support)
                deformations = deformations * (1 - left_edge) * (1 - right_edge)
            case 'nonboundary_handle_fixed':
                center_point = torch.exp(-((curvilinear_coords[...,0:1] - 0.5 * self.curvilinear_space.xi__1_max) ** 2 + (curvilinear_coords[...,1:2] - 0.7 * self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)
                deformations = deformations * (1 - center_point)
            case 'nonboundary_edge_fixed':
                center_edge = torch.exp(-((curvilinear_coords[...,0:1] - 0.5 * self.curvilinear_space.xi__1_max) ** 2)/self.boundary_support)
                deformations = deformations * (1 - center_edge)
            case 'top_bottom_rims_compression':                
                bottom_rim = torch.exp(-(curvilinear_coords[...,1:2] ** 2)/self.boundary_support)
                top_rim = torch.exp(-((curvilinear_coords[...,1:2] - self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)
                temporal_motion = 0.075 * torch.ones_like(curvilinear_coords[...,0:1])
                rim_displacement = torch.cat([torch.zeros_like(temporal_motion), temporal_motion, torch.zeros_like(temporal_motion)], dim=2)
                deformations = deformations * (1 - bottom_rim) * (1 - top_rim) - rim_displacement * top_rim + rim_displacement * bottom_rim
            case 'top_bottom_rims_torsion':
                self.boundary_support = 0.001
                bottom_rim = torch.exp(-(curvilinear_coords[...,1:2] ** 2)/self.boundary_support)
                top_rim = torch.exp(-((curvilinear_coords[...,1:2] - self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)
                R = 0.25
                rotation = math.pi / 4 #3 * math.pi / 4
                temporal_motion = torch.ones_like(curvilinear_coords[...,0:1]) * rotation
                top_rim_displacement = torch.cat([R * (torch.cos(curvilinear_coords[...,0:1] + temporal_motion) - torch.cos(curvilinear_coords[...,0:1])), torch.zeros_like(temporal_motion), R * (torch.sin(curvilinear_coords[...,0:1] + temporal_motion) - torch.sin(curvilinear_coords[...,0:1]))], dim=2)
                bottom_rim_displacement = torch.cat([R * (torch.cos(curvilinear_coords[...,0:1] - temporal_motion) - torch.cos(curvilinear_coords[...,0:1])), torch.zeros_like(temporal_motion), R * (torch.sin(curvilinear_coords[...,0:1] - temporal_motion) - torch.sin(curvilinear_coords[...,0:1])), ], dim=2)                
                deformations = deformations * (1 - bottom_rim) * (1 - top_rim) + top_rim_displacement * top_rim + bottom_rim_displacement * bottom_rim              
            case 'top_rim_fixed':
                top_rim = torch.exp(-((curvilinear_coords[...,1:2] - self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)
                deformations = deformations * (1 - top_rim)
            case 'top_rim_torsion':
                top_rim = torch.exp(-((curvilinear_coords[...,1:2] - self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)
                R_top = 0.2
                rotation = math.pi / 2
                temporal_motion = torch.ones_like(curvilinear_coords[...,0:1]) * rotation
                top_rim_displacement = torch.cat([R_top * (torch.cos(curvilinear_coords[...,0:1] + temporal_motion) - torch.cos(curvilinear_coords[...,0:1])), torch.zeros_like(temporal_motion), R_top * (torch.sin(curvilinear_coords[...,0:1] + temporal_motion) - torch.sin(curvilinear_coords[...,0:1]))], dim=2)
                deformations = deformations * (1 - top_rim) + top_rim_displacement * top_rim            
            case 'mesh_vertices':
                for i in range(self.boundary_curvilinear_coords.shape[0]):
                    boundary_point = torch.exp(-((curvilinear_coords[...,0:1] - self.boundary_curvilinear_coords[i][0]) ** 2 + (curvilinear_coords[...,1:2] - self.boundary_curvilinear_coords[i][1]) ** 2)/self.boundary_support)
                    deformations = deformations * (1 - boundary_point)
                deformations = deformations
            case _:
                raise ValueError(f'Unknown boundary condition: {self.boundary_condition_name}')
        return deformations
                        
def test(ndf: Siren, reference_midsurface: ReferenceMidSurface, meshes_dir: str, i: int):
    
    test_deformations = ndf(reference_midsurface.template_mesh.textures.verts_uvs_padded())
    test_deformed_positions = reference_midsurface.template_mesh.verts_padded() + test_deformations
    if tb.writer:
        tb.writer.add_mesh('simulated_states', test_deformed_positions, faces=reference_midsurface.template_mesh.textures.faces_uvs_padded(), global_step=i)
    save_meshes(test_deformed_positions, reference_midsurface.template_mesh.textures.faces_uvs_padded()[0], meshes_dir, i, reference_midsurface.template_mesh.textures.verts_uvs_padded()[0]) 
        
def train():  
    args = get_config_parser().parse_args()
    log_dir = os.path.join(args.logging_dir, args.expt_name)
    meshes_dir = os.path.join(log_dir, 'meshes')   
    weights_dir = os.path.join(log_dir, 'weights')
        
    for dir in [log_dir, weights_dir]:
        os.makedirs(dir, exist_ok=True)
            
    logger = get_logger(log_dir, args.expt_name)
    logger.info(args)
    tb.set_tensorboard_writer(log_dir, args.debug)
    copyfile(args.config_filepath, os.path.join(log_dir, 'args.ini'))

    curvilinear_space = CurvilinearSpace(args.xi__1_max, args.xi__2_max)
    reference_midsurface = ReferenceMidSurface(args, curvilinear_space)
    boundary = Boundary(args.reference_geometry_name, args.boundary_condition_name, curvilinear_space, reference_midsurface.boundary_curvilinear_coords)
    
    ndf = Siren(boundary, in_features=3 if args.reference_geometry_name in ['cylinder', 'cone'] else 2).to(device)
    optimizer = torch.optim.Adam(lr=args.lrate, params=ndf.parameters())
    
    if args.i_ckpt is not None:
        ckpts = [os.path.join(weights_dir, f'{args.i_ckpt:06d}.tar')]
    else:
        ckpts = [os.path.join(weights_dir, f) for f in sorted(os.listdir(weights_dir)) if '0.tar' in f]
                
    logger.info(f'Found ckpts: {ckpts}')
    if len(ckpts) > 0 and not args.no_reload:
        logger.info(f'Resuming experiment {args.expt_name} from checkpoint {ckpts[-1]}')
        ckpt = torch.load(ckpts[-1])
        global_step = ckpt['global_step']
        ndf.load_state_dict(ckpt['siren_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    else: 
        logger.info(f'Starting experiment {args.expt_name}')
        global_step = 0
    
    if args.test_only:
        logger.info(f'Evaluating NDF from checkpoint {ckpts[-1]}')
        test(ndf, reference_midsurface, meshes_dir, global_step)
        return
    
    reference_geometry = ReferenceGeometry(reference_midsurface, args.train_n_spatial_samples)
    material = LinearMaterial(args, reference_geometry) if args.material_type == 'linear' else NonLinearMaterial(args, reference_geometry)
    energy = Energy(reference_geometry, material, args.gravity_acceleration, args.i_debug)
    
    if args.reference_geometry_name == 'mesh':
        sampler = MeshSampler(args.train_n_spatial_samples, reference_midsurface.template_mesh)
    else:
        sampler = GridSampler(args.train_n_spatial_samples, curvilinear_space)
    dataloader = DataLoader(sampler, batch_size=1, num_workers=0)
    
    if tb.writer:
        tb.writer.add_text('args', str(args))
        
    for i in trange(global_step, args.n_iterations):
        curvilinear_coords = next(iter(dataloader))
        reference_geometry(curvilinear_coords, i==0)
        deformations = ndf(reference_geometry.curvilinear_coords)                    
        mechanical_energy = energy(deformations, i)
        loss = mechanical_energy.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        if tb.writer:               
            tb.writer.add_scalar('loss/loss', loss, i)
            tb.writer.add_scalar('param/mean_deformation', deformations.mean(), i)
            
        if args.decay_lrate:
            new_lrate = args.lrate * args.lrate_decay_rate ** (i / args.lrate_decay_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate         
        
        if not i % args.i_summary:
            logger.info(f'Iteration: {i}, loss: {loss}, mean_deformation: {deformations.mean()}')      
            
        if not i % args.i_weights and i > 0:
            torch.save({
                'global_step': i,
                'siren_state_dict': ndf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(weights_dir, f'{i:06d}.tar'))
            
        if not i % args.i_test:            
            test(ndf, reference_midsurface, meshes_dir, i)
    tb.writer.flush()
    tb.writer.close()

if __name__=='__main__':
    train()