'''
python train.py -c config/napkin.ini --expt_name napkin -m material/canvas.ini
python train.py -c config/napkin_mesh.ini --expt_name napkin_mesh
python train.py -c config/flag_mesh.ini --expt_name flag_mesh_vis_tangents
python train.py -c config/drape.ini -n drape_nl_canvas -m material/canvas.ini
python train.py -c config/sleeve_buckle.ini -n sleeve_buckle_canvas -m material/canvas.ini
python train.py -c config/skirt_twist.ini -n skirt_twist -m material/linear_1.ini
python train.py -c config/skirt_static_rim.ini -n skirt_static_rim -m material/canvas.ini
python train.py -c config/collision.ini -n collision_linear -m material/linear_1.ini
tensorboard --logdir ? --port=?
'''
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
from diff_operators import jacobian
from internal_energy import compute_energy
from test import test
from logger import get_logger
from config_parser import get_config_parser
from config import device
from reference_midsurface import ReferenceMidSurface
from modules import compute_sdf
from helper import get_plot_single_tensor
    
def train():  
    relu = nn.ReLU()
    eps = 0.00#0.001
    args = get_config_parser().parse_args()
    log_dir = os.path.join(args.logging_dir, args.expt_name)
    meshes_dir = os.path.join(log_dir, 'meshes')   
    images_dir = os.path.join(log_dir, 'images')
    weights_dir = os.path.join(log_dir, 'weights')
        
    for dir in [log_dir, weights_dir]: #[log_dir, meshes_dir, images_dir, weights_dir]
        os.makedirs(dir, exist_ok=True)
    
    tb_writer = SummaryWriter(log_dir)
    logger = get_logger(log_dir, args.expt_name)
    logger.info(args)
    copyfile(args.config_filepath, os.path.join(log_dir, 'args.ini'))
     
    reference_midsurface = ReferenceMidSurface(args, tb_writer)
    
    ndf = Siren(args.end_time, args.xi__1_scale, args.xi__2_scale, args.boundary_condition_name, args.reference_geometry_name, boundary_curvilinear_coords=reference_midsurface.boundary_curvilinear_coords).to(device)
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
        test(ndf, args.test_temporal_sidelen, meshes_dir, images_dir, f'{global_step}', args.texture_image_file, reference_midsurface, tb_writer)
        return
    
    if args.material_type == 'linear':            
        material = LinearMaterial(args)
    elif args.material_type == 'nonlinear':
        material = NonLinearMaterial(args)
    external_load = torch.tensor(args.gravity_acceleration, device=device).expand(1, args.train_temporal_sidelen * args.train_spatial_sidelen**2, 3) * material.mass_area_density #* material.thickness

    if False: #args.reference_geometry_name in ['mesh']:
        sampler = MeshSampler(reference_midsurface.template_mesh, reference_midsurface.curvilinear_coords, args.train_spatial_sidelen, args.train_temporal_sidelen, args.end_time)
    else:
        sampler = GridSampler(args.train_spatial_sidelen, args.train_temporal_sidelen, args.end_time, args.xi__1_scale, args.xi__2_scale, 'train')
    dataloader = DataLoader(sampler, batch_size=1, num_workers=0)
    
    tb_writer.add_text('args', str(args))
    for i in trange(global_step, args.n_iterations):
        curvilinear_coords, temporal_coords = next(iter(dataloader))
        ref_geometry = ReferenceGeometry(curvilinear_coords, args.train_spatial_sidelen, args.train_temporal_sidelen, reference_midsurface, tb_writer, debug_ref_geometry=False)
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
            test(ndf, args.test_temporal_sidelen, meshes_dir, images_dir, i, args.texture_image_file, reference_midsurface, tb_writer)
    tb_writer.flush()
    tb_writer.close()

if __name__=='__main__':
    train()