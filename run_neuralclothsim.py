import torch
import os
from tqdm import trange
from shutil import copyfile
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from material import LinearMaterial, NonLinearMaterial
from sampler import GridSampler, MeshSampler
from reference_geometry import ReferenceGeometry
from modules import Siren
from internal_energy import compute_energy
from logger import get_logger
from config_parser import get_config_parser
from config_parser import device
from reference_midsurface import ReferenceMidSurface
from modules import compute_sdf
from file_io import save_meshes
#torch.manual_seed(2) #Set seed for reproducible results

def test(ndf, test_temporal_sidelen, meshes_dir, i, reference_midsurface, tb_writer):
    
    #test_deformations = ndf(reference_midsurface.curvilinear_coords.repeat(1, test_temporal_sidelen, 1), reference_midsurface.mesh_temporal_coords)
    test_deformations = ndf(reference_midsurface.curvilinear_coords.repeat(1, test_temporal_sidelen, 1), reference_midsurface.temporal_coords)
    test_deformed_positions = reference_midsurface.vertices.repeat(1, test_temporal_sidelen, 1) + test_deformations
    '''
    sdf, normal = compute_sdf(test_deformed_positions)
    eps = 0.0
    outward_displacement = torch.einsum('ij,ijk->ijk', relu(eps - sdf), normal)#relunn.GELU()
    test_deformed_positions = test_deformed_positions + outward_displacement
    '''
    #test_deformed_positions = reference_midsurface.midsurface(reference_midsurface.curvilinear_coords).repeat(1, test_temporal_sidelen, 1) + test_deformations
    tb_writer.add_mesh('simulated_states', test_deformed_positions.view(test_temporal_sidelen, -1, 3), faces=reference_midsurface.faces.repeat(test_temporal_sidelen, 1, 1), global_step=i)
    save_meshes(test_deformed_positions, reference_midsurface.faces, meshes_dir, i, test_temporal_sidelen, reference_midsurface.curvilinear_coords) 
        
def train():  
    relu = nn.ReLU()
    eps = 0.00#0.001
    args = get_config_parser().parse_args()
    log_dir = os.path.join(args.logging_dir, args.expt_name)
    meshes_dir = os.path.join(log_dir, 'meshes')   
    weights_dir = os.path.join(log_dir, 'weights')
        
    for dir in [log_dir, weights_dir]:
        os.makedirs(dir, exist_ok=True)
    
    tb_writer = SummaryWriter(log_dir)
    logger = get_logger(log_dir, args.expt_name)
    logger.info(args)
    copyfile(args.config_filepath, os.path.join(log_dir, 'args.ini'))
     
    reference_midsurface = ReferenceMidSurface(args, tb_writer)
    
    ndf = Siren(args.xi__1_scale, args.xi__2_scale, args.boundary_condition_name, args.reference_geometry_name, boundary_curvilinear_coords=reference_midsurface.boundary_curvilinear_coords).to(device)
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
        test(ndf, args.test_temporal_sidelen, meshes_dir, f'{global_step}', reference_midsurface, tb_writer)
        return
    
    if args.material_type == 'linear':            
        material = LinearMaterial(args)
    elif args.material_type == 'nonlinear':
        material = NonLinearMaterial(args)
    external_load = torch.tensor(args.gravity_acceleration, device=device).expand(1, args.train_temporal_sidelen * args.train_spatial_sidelen**2, 3) * material.mass_area_density #* material.thickness

    if False: #args.reference_geometry_name in ['mesh']:
        sampler = MeshSampler(reference_midsurface.template_mesh, reference_midsurface.curvilinear_coords, args.train_spatial_sidelen, args.train_temporal_sidelen)
    else:
        sampler = GridSampler(args.train_spatial_sidelen, args.train_temporal_sidelen, args.xi__1_scale, args.xi__2_scale, 'train')
    dataloader = DataLoader(sampler, batch_size=1, num_workers=0)
    
    tb_writer.add_text('args', str(args))
    for i in trange(global_step, args.n_iterations):
        curvilinear_coords, temporal_coords = next(iter(dataloader))
        ref_geometry = ReferenceGeometry(curvilinear_coords, args.train_spatial_sidelen, args.train_temporal_sidelen, reference_midsurface, tb_writer, debug_ref_geometry=True)
        deformations = ndf(ref_geometry.curvilinear_coords, temporal_coords)                    

        collision_loss = torch.tensor(0., device=device)
        mechanical_energy = compute_energy(deformations, ref_geometry, material, external_load, temporal_coords, tb_writer, i)
        physics_loss = mechanical_energy.mean()

        loss = physics_loss + collision_loss
        #loss = physics_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        with torch.no_grad():
            #if i == 0:
            #    tb_writer.add_graph(ndf, (ref_geometry.curvilinear_coords, temporal_coords))
            tb_writer.add_scalar('loss/total_loss', loss, i)
            tb_writer.add_scalar('loss/physics_loss', physics_loss, i)
            tb_writer.add_scalar('loss/collision_loss', collision_loss, i)
            tb_writer.add_scalar('param/mean_deformation', deformations.mean(), i)  
        if args.decay_lrate:
            new_lrate = args.lrate * args.lrate_decay_rate ** (i / args.lrate_decay_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate         
        
        if not i % args.i_summary:
            logger.info(f'Iteration: {i}, physics_loss: {physics_loss}, mean_deformation: {deformations.mean()}')
            logger.info(f'Iteration: {i}, collision_loss: {collision_loss}')      
            
        if not i % args.i_weights and i > 0:
            torch.save({
                'global_step': i,
                'siren_state_dict': ndf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(weights_dir, f'{i:06d}.tar'))
            
        if not i % args.i_test:            
            test(ndf, args.test_temporal_sidelen, meshes_dir, i, reference_midsurface, tb_writer)
    tb_writer.flush()
    tb_writer.close()

if __name__=='__main__':
    train()