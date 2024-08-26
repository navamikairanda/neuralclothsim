import torch
import numpy as np
from torch.utils.data import Dataset
from config_parser import device

# Parts of the code borrowed from Meta Platforms, Inc.
from typing import Tuple, Union, Literal
from pytorch3d.structures import Meshes
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.packed_to_padded import packed_to_padded

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
    
    # Use the barycentric coords to get the uv coodinate for the samepled point.
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
                       
class GridSampler(Dataset):
    def __init__(self, spatial_sidelen: int, temporal_sidelen: int, xi__1_scale: float, xi__2_scale: float, mode: Literal['train', 'test']):
        super().__init__()
        self.mode = mode
        self.spatial_sidelen = spatial_sidelen
        self.temporal_sidelen = temporal_sidelen
                
        self.xi__1_scale = xi__1_scale
        self.xi__2_scale = xi__2_scale
        if self.mode == 'train':
            self.cell_temporal_coords = get_mgrid((self.temporal_sidelen,), stratified=True, dim=1)
            self.cell_curvilinear_coords = get_mgrid((self.spatial_sidelen, self.spatial_sidelen), stratified=True, dim=2)
        elif self.mode == 'test':
            self.node_temporal_coords = get_mgrid((self.temporal_sidelen,), stratified=False, dim=1)
            self.node_curvilinear_coords = get_mgrid((self.spatial_sidelen, self.spatial_sidelen), stratified=False, dim=2)
            
    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
        if self.mode == 'test': 
            curvilinear_coords = self.node_curvilinear_coords.clone()
            temporal_coords = self.node_temporal_coords.clone() 
            curvilinear_coords[...,0] *= self.xi__1_scale 
            curvilinear_coords[...,1] *= self.xi__2_scale
        elif self.mode == 'train':            
            curvilinear_coords = self.cell_curvilinear_coords.clone() 
            temporal_coords = self.cell_temporal_coords.clone() 
            
            t_rand_temporal = torch.rand([self.temporal_sidelen, 1], device=device) / self.temporal_sidelen
            t_rand_spatial = torch.rand([self.spatial_sidelen**2, 2], device=device) / self.spatial_sidelen
            temporal_coords += t_rand_temporal
            curvilinear_coords += t_rand_spatial
            
            curvilinear_coords[...,0] *= self.xi__1_scale
            curvilinear_coords[...,1] *= self.xi__2_scale
            curvilinear_coords.requires_grad_(True)
            temporal_coords.requires_grad_(True)
            
        temporal_coords = temporal_coords.repeat_interleave(self.spatial_sidelen**2, 0)        
        return curvilinear_coords, temporal_coords            

class MeshSampler(Dataset):
    def __init__(self, reference_mesh: Meshes, n_samples: int, temporal_sidelen: int):
        super().__init__()
        self.reference_mesh = reference_mesh
        self.n_samples = n_samples
        self.temporal_sidelen = temporal_sidelen
                
        self.cell_temporal_coords = get_mgrid((self.temporal_sidelen,), stratified=True, dim=1)
            
    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError

        curvilinear_coords = sample_points_from_meshes(self.reference_mesh, self.n_samples)[1][0]
        
        temporal_coords = self.cell_temporal_coords.clone() 
        
        t_rand_temporal = torch.rand([self.temporal_sidelen, 1], device=device) / self.temporal_sidelen
        temporal_coords += t_rand_temporal
        
        curvilinear_coords.requires_grad_(True)
        temporal_coords.requires_grad_(True)
            
        temporal_coords = temporal_coords.repeat_interleave(self.n_samples, 0)        
        return curvilinear_coords, temporal_coords            