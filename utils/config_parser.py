import math
import configargparse
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_config_parser():
    parser = configargparse.ArgumentParser()
    parser.add('-c', '--config_filepath', required=True, is_config_file=True, help='config file path')
    parser.add_argument('-n', '--expt_name', type=str, required=True, help='experiment name; this will also be the name of subdirectory in logging_dir')

    # simulation parameters
    parser.add_argument('--reference_geometry_name', type=str, default='rectangle_xy', help='name of the reference geometry; can be rectangle_xy, rectangle_xz, cylinder, cone or mesh')
    parser.add_argument('--xi__1_max', type=float, default=2 * math.pi, help='max value for xi__1; 2 * pi for cylinder or cone, and Lx for rectangle. min value for xi__1 is assumed to be 0')
    parser.add_argument('--xi__2_max', type=float, default=1, help='max value for xi__2; Ly for all reference geometries. min value for xi__2 is assumed to be 0')
    parser.add_argument('--boundary_condition_name', type=str, default='top_left_fixed', help='name of the spatio-temporal boundary condition; can be one of top_left_fixed, top_left_top_right_moved, adjacent_edges_fixed, nonboundary_handle_fixed, nonboundary_edge_fixed for reference geometry as a rectangle, and top_bottom_rims_compression, top_bottom_rims_torsion for cylinder, top_rim_fixed, top_rim_torsion for cone, and mesh_vertices for mesh')
    parser.add_argument('--gravity_acceleration', type=float, nargs='+', default=[0,-9.8, 0], help='acceleration due to gravity')
    parser.add_argument('--trajectory', action='store_true', help='whether to simulate the trajectory from the reference state to the quasi-static state; otherwise, the quasistatic solutions are computed at all temporal samples')
    
    # additional parameters if the reference geometry is a mesh
    parser.add_argument('--reference_geometry_source', type=str, default='assets/textured_uniform_1_020.obj', help='source file for reference geometry')
    parser.add_argument('--reference_mlp_n_iterations', type=int, default=3000, help='number of iterations for fitting the reference geometry MLP')
    parser.add_argument('--reference_mlp_lrate', type=float, default=5e-6, help='learning rate for the reference geometry MLP')
    parser.add_argument('--reference_boundary_vertices', type=int, nargs='+', help='vertices for boundary condition on the reference geometry')
    
    # material parameters
    parser.add('-m', '--material_filepath', is_config_file=True, default='material/linear_1.ini', help='name of the material')
    parser.add_argument('--material_type', type=str, help='type of material; can be linear (isotropic) or nonlinear (orthotropic Clyde model)')
    parser.add_argument('--StVK', action='store_true', help='whether to use the St.Venant-Kirchhoff simplification of the Clyde material model')
    
    parser.add_argument('--mass_area_density', type=float, default=0.144, help='mass area density in kg/m^2 ')
    parser.add_argument('--thickness', type=float, default=0.0012, help='thickness in meters')
    
    # material parameters for a linear material
    parser.add_argument('--youngs_modulus', type=float, default=5000, help='Young\'s modulus')
    parser.add_argument('--poissons_ratio', type=float, default=0.25, help='Poisson\'s ratio')

    # material parameters for a nonlinear material [Clyde et al., 2017]
    parser.add_argument('--a11', type=float, help='a11')
    parser.add_argument('--a12', type=float, help='a12')
    parser.add_argument('--a22', type=float, help='a22')
    parser.add_argument('--G12', type=float, help='G12')
        
    parser.add_argument('--d', type=int, nargs='+', help='degree')
    parser.add_argument('--mu1', type=float, nargs='+', help='mu1')
    parser.add_argument('--mu2', type=float, nargs='+', help='mu2')
    parser.add_argument('--mu3', type=float, nargs='+', help='mu3')
    parser.add_argument('--mu4', type=float, nargs='+', help='mu4')
    parser.add_argument('--alpha1', type=float, nargs='+', help='alpha1')
    parser.add_argument('--alpha2', type=float, nargs='+', help='alpha2')
    parser.add_argument('--alpha3', type=float, nargs='+', help='alpha3')
    parser.add_argument('--alpha4', type=float, nargs='+', help='alpha4')
    
    parser.add_argument('--E_11_min', type=float, help='E_11_min')
    parser.add_argument('--E_11_max', type=float, help='E_11_max')
    parser.add_argument('--E_22_min', type=float, help='E_22_min')
    parser.add_argument('--E_22_max', type=float, help='E_22_max')
    parser.add_argument('--E_12_max', type=float, help='E_12_max')
    
    # training options
    parser.add_argument('--train_n_spatial_samples', type=int, default=400, help='N_omega, number of samples used for training; when the reference geometry is an analytical surface, the number of spatial grid samples along each curvilinear coordinate is square_root(N_omega)')
    parser.add_argument('--train_n_temporal_samples', type=int, default=10, help='N_t, number of temporal samples used for training')
    parser.add_argument('--lrate', type=float, default=5e-6, help='learning rate')
    parser.add_argument('--decay_lrate', action='store_true', default=True, help='whether to decay learning rate')
    parser.add_argument('--lrate_decay_steps', type=int, default=5000, help='learning rate decay steps')
    parser.add_argument('--lrate_decay_rate', type=float, default=0.1, help='learning rate decay rate')    
    parser.add_argument('--n_iterations', type=int, default=5001, help='total number of training iterations')
    
    # logging/saving options
    parser.add_argument('--logging_dir', type=str, default='logs', help='root directory for logging')
    parser.add_argument("--i_weights", type=int, default=400, help='frequency of saving NDF weights as checkpoints')
    parser.add_argument("--i_summary", type=int, default=100, help='frequency of logging losses')
    parser.add_argument("--i_test", type=int, default=100, help='frequency of evaluating NDF and saving simulated meshes during training')
    
    # debug options
    parser.add_argument('--debug', action='store_true', default=True, help='whether to run in debug mode; this will log the reference geometric quantities (e.g. metric and curvature tensor), the strains, and the simulated states to TensorBoard')
    parser.add_argument('--i_debug', type=int, default=200, help='frequency of Tensordboard logging')
    
    # reload options
    parser.add_argument('--no_reload', action='store_true', help='do not resume training from checkpoint, rather train from scratch')
    parser.add_argument("--i_ckpt", type=int, help='weight checkpoint to reload for resuming training or performing evaluation; if None, the latest checkpoint is used')
    parser.add_argument('--test_only', action='store_true', help='evaluate NDF from i_ckpt and save simulated meshes; do not resume training')
    
    # testing options
    parser.add_argument('--test_n_spatial_samples', type=int, default=400, help='N_omega, the number of samples used for evaluation when reference geometry is an analytical surface; if the reference geometry is instead a mesh, this argument is ignored, and the samples (vertices and faces) used for evaluation will match that of template mesh') 
    parser.add_argument('--test_n_temporal_samples', type=int, default=2, help='N_t, number of temporal samples used for evaluation')
                            
    return parser