import torch
import os
import math
from tqdm import trange
from shutil import copyfile

from torch.utils.data import DataLoader
import tb
from material import LinearMaterial, NonLinearMaterial
from sampler import GridSampler, MeshSampler, CurvilinearSpace
from reference_geometry import ReferenceGeometry
from modules import Siren
from energy import Energy
from logger import get_logger
from config_parser import get_config_parser, device
from reference_midsurface import ReferenceMidSurface
from boundary import Boundary
from file_io import save_meshes
#torch.manual_seed(2) #Set seed for reproducible results

def test(ndf: Siren, test_n_temporal_samples: int, meshes_dir: str, i: int, reference_midsurface: ReferenceMidSurface):
    
    test_deformations = ndf(reference_midsurface.template_mesh.textures.verts_uvs_padded()[0].repeat(1, test_n_temporal_samples, 1), reference_midsurface.temporal_coords)
    test_deformed_positions = reference_midsurface.template_mesh.verts_padded().repeat(1, test_n_temporal_samples, 1) + test_deformations
    if tb.writer:
        tb.writer.add_mesh('simulated_states', test_deformed_positions.view(test_n_temporal_samples, -1, 3), faces=reference_midsurface.template_mesh.textures.faces_uvs_padded().repeat(test_n_temporal_samples, 1, 1), global_step=i)
    save_meshes(test_deformed_positions, reference_midsurface.template_mesh.textures.faces_uvs_padded()[0], meshes_dir, i, test_n_temporal_samples, reference_midsurface.template_mesh.textures.verts_uvs_padded()[0]) 
        
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

    if args.reference_geometry_name != 'mesh': # for analytical surface, use equal number of samples along each curvilinear coordinate
        args.train_n_spatial_samples, args.test_n_spatial_samples = math.isqrt(args.train_n_spatial_samples) ** 2, math.isqrt(args.test_n_spatial_samples) ** 2
    
    curvilinear_space = CurvilinearSpace(args.xi__1_max, args.xi__2_max)
    reference_midsurface = ReferenceMidSurface(args, curvilinear_space)
    reference_geometry = ReferenceGeometry(args.train_n_spatial_samples, args.train_n_temporal_samples, reference_midsurface)
    boundary = Boundary(args.reference_geometry_name, args.boundary_condition_name, curvilinear_space, reference_midsurface.boundary_curvilinear_coords)
    
    ndf = Siren(boundary, in_features=4 if args.reference_geometry_name in ['cylinder', 'cone'] else 3).to(device)
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
        test(ndf, args.test_n_temporal_samples, meshes_dir, f'{global_step}', reference_midsurface)
        return
    
    material = LinearMaterial(args) if args.material_type == 'linear' else NonLinearMaterial(args)    
    external_load = torch.tensor(args.gravity_acceleration, device=device) * material.mass_area_density
    external_load = external_load.expand(1, args.train_n_temporal_samples * args.train_n_spatial_samples, 3)
    energy = Energy(reference_geometry, material, external_load, args.i_debug)
    
    if args.reference_geometry_name == 'mesh':
        sampler = MeshSampler(args.train_n_spatial_samples, args.train_n_temporal_samples, reference_midsurface.template_mesh)
    else:
        sampler = GridSampler(args.train_n_spatial_samples, args.train_n_temporal_samples, curvilinear_space)

    dataloader = DataLoader(sampler, batch_size=1, num_workers=0)
    
    if tb.writer:
        tb.writer.add_text('args', str(args))
        
    for i in trange(global_step, args.n_iterations):
        curvilinear_coords, temporal_coords = next(iter(dataloader))
        reference_geometry(curvilinear_coords, i==0)
        deformations = ndf(reference_geometry.curvilinear_coords, temporal_coords)                    
        mechanical_energy = energy(deformations, temporal_coords, i)
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
            test(ndf, args.test_n_temporal_samples, meshes_dir, i, reference_midsurface)
    tb.writer.flush()
    tb.writer.close()

if __name__=='__main__':
    train()