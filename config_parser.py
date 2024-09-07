import math
import configargparse
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_config_parser():
    parser = configargparse.ArgumentParser()
    parser.add('-c', '--config_filepath', required=True, is_config_file=True, help='config file path')
    parser.add_argument('-n', '--expt_name', type=str, required=True, help='experiment name; also the name of subdirectory in logging_dir')

    # simulation parameters
    parser.add_argument('--reference_geometry_name', type=str, required=True, help='name of the reference geometry; can be rectangle_xy, rectangle_xz,  cylinder, cone or mesh')
    parser.add_argument('--xi__1_max', type=float, default=2 * math.pi, help='max value for xi__1; 2 * pi for cylinder or cone, and Lx for rectangle. min value for xi__1 is assumed to be 0')
    parser.add_argument('--xi__2_max', type=float, default=1, help='max value for xi__2; Ly for all reference geometries. min value for xi__2 is assumed to be 0')
    parser.add_argument('--boundary_condition_name', type=str, help='name of the spatio-temporal boundary condition; can be top_left_fixed, top_left_top_right_drape for rectangle, and two_rims_compression for cylinder, and top_rim_fixed, top_rim_torsion for cone')
    parser.add_argument('--gravity_acceleration', type=float, nargs='+', default=[0,-9.8, 0], help='acceleration due to gravity')
    
    # The following parameters are relevant only if the reference geometry is a mesh (i.e. reference_geometry_name == 'mesh')
    parser.add_argument('--reference_geometry_source', type=str, default='assets/meshes/flag.obj', help='source file for reference geometry')
    parser.add_argument('--reference_mlp_n_iterations', type=int, default=5000, help='number of iterations for fitting the reference geometry MLP')
    parser.add_argument('--reference_mlp_lrate', type=float, default=1e-5, help='learning rate for the reference geometry MLP')
    parser.add_argument('--boundary_condition_vertices', type=int, nargs='+', help='vertices for boundary condition on the reference geometry')
    
    # material parameters
    parser.add('-m', '--material_filepath', is_config_file=True, default='material/canvas.ini', help='name of the material')
    parser.add_argument('--material_type', type=str, help='type of material; can be linear (isotropic) or nonlinear (orthotropic Clyde model)')
    parser.add_argument('--StVK', action='store_true', help='whether to use the St.Venant-Kirchhoff simplification of the Clyde material model')
    
    parser.add_argument('--mass_area_density', type=float, default=0.144, help='mass area density or rho in kg/m^2 ')
    parser.add_argument('--thickness', type=float, default=0.0012, help='thickness or tau in meters')
    
    parser.add_argument('--youngs_modulus', type=float, default=9000, help='Young\'s modulus')
    parser.add_argument('--poissons_ratio', type=float, default=0.25, help='Poisson\'s ratio')

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
    
    # transition point to extrapolation
    parser.add_argument('--E_11_min', type=float, help='E_11_min')
    parser.add_argument('--E_11_max', type=float, help='E_11_max')
    parser.add_argument('--E_22_min', type=float, help='E_22_min')
    parser.add_argument('--E_22_max', type=float, help='E_22_max')
    parser.add_argument('--E_12_max', type=float, help='E_12_max')
    
    # training options
    #parser.add_argument('--train_spatial_sidelen', type=int, default=20, help='square_root(N_omega), number of spatial grid samples along each curvilinear coordinate used for training when reference geometry is an analytical surface')
    #parser.add_argument('--train_n_mesh_samples', type=int, default=100, help='number of samples used for training when reference geometry is a mesh')
    parser.add_argument('--train_n_spatial_samples', type=int, default=100, help='N_omega, number of samples used for training; when reference geometry is an analytical surface, number of spatial grid samples along each curvilinear coordinate is square_root(N_omega)')
    parser.add_argument('--train_temporal_sidelen', type=int, default=20, help='N_t, number of temporal samples, at training')
    parser.add_argument('--lrate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--decay_lrate', action='store_true', help='whether to decay learning rate')
    parser.add_argument('--lrate_decay_steps', type=int, default=10000, help='learning rate decay steps')
    parser.add_argument('--lrate_decay_rate', type=float, default=0.1, help='learning rate decay rate')    
    parser.add_argument('--n_iterations', type=int, default=10000, help='number of training iterations')
    parser.add_argument('--debug', action='store_true', help='whether to run NeuralClothSim in debug mode; this will plot the reference geometric quantities (metric and curvature tensor) and the strains due to deformation (membrane, bending, orthotropic, and strain energy density) to TensorBoard')
    
    # logging/saving options
    parser.add_argument('--logging_dir', type=str, default='logs', help='root directory for logging')
    parser.add_argument("--i_weights", type=int, default=200, help='frequency of saving NDF weights as checkpoints')
    parser.add_argument("--i_summary", type=int, default=100, help='frequency of plotting losses')
    parser.add_argument("--i_test", type=int, default=100, help='frequency of evaluating NDF and saving resulting meshes and videos')
    parser.add_argument('--no_reload', action='store_true', help='do not resume training from checkpoint')
    parser.add_argument('--siren_omega_0', type=float, default=30, help='omega_0 for SIREN')
    
    # testing options
    parser.add_argument('--test_only', action='store_true', help='evaluate NDF from i_ckpt and save resulting meshes and videos')
    parser.add_argument('--test_n_spatial_samples', type=int, default=400, help='N_omega, the number of samples used for evaluation when reference geometry is an analytical surface; if the reference geometry is instead a mesh, this argument is ignored, and the samples used for evaluation will match the vertices in the mesh') 
    # when reference geometry is a mesh, the same vertices and faces are used for evaluation as the input mesh
    parser.add_argument('--test_temporal_sidelen', type=int, default=20, help='N_t, number of temporal samples, at training')
    
    # reload options
    parser.add_argument("--i_ckpt", type=int, help='weight checkpoint to reload')
 
    # evaluation options
    parser.add_argument('--i_converged', type=str, help='iteration at which the reconstruction is considered to have converged')
                            
    return parser