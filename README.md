# NeuralClothSim: Neural Deformation Fields Meet the Thin Shell Theory
### [Project Page](https://4dqv.mpi-inf.mpg.de/NeuralClothSim/) | [Paper](https://arxiv.org/pdf/2308.12970) | [Video](https://www.youtube.com/watch?v=z-7MBiAi7SM) 
[![Explore NeuralClothSim in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/navamikairanda/neuralclothsim/blob/main/neuralclothsim.ipynb)<br>

[Navami Kairanda](https://people.mpi-inf.mpg.de/~nkairand/),
[Marc Habermann](https://people.mpi-inf.mpg.de/~mhaberma/),
[Christian Theobalt](https://people.mpi-inf.mpg.de/~theobalt/),
[Vladislav Golyanik](https://people.mpi-inf.mpg.de/~golyanik/) <br>
Max Planck Institute for Informatics <br>

This repository contains the official implementation of the paper "NeuralClothSim: Neural Deformation Fields Meet the Thin Shell Theory".

[<img src="https://i3.ytimg.com/vi/z-7MBiAi7SM/maxresdefault.jpg" width="500">](https://www.youtube.com/watch?v=z-7MBiAi7SM)

## What is NeuralClothSim?
*Despite existing 3D cloth simulators producing realistic results, they predominantly operate on discrete surface representations (e.g. points and meshes) with a fixed spatial resolution, which often leads to large memory consumption and resolution-dependent simulations. Moreover, back-propagating gradients through the existing solvers is difficult, and they hence cannot be easily integrated into modern neural architectures. In response, this paper re-thinks physically plausible cloth simulation: We propose NeuralClothSim, i.e., a new quasistatic cloth simulator using thin shells, in which surface deformation is encoded in neural network weights in the form of a neural field. Our memory-efficient solver operates on a new continuous coordinate-based surface representation called neural deformation fields (NDFs); it supervises NDF equilibria with the laws of the non-linear Kirchhoff-Love shell theory with a non-linear anisotropic material model. NDFs are adaptive: They 1) allocate their capacity to the deformation details and 2) allow surface state queries at arbitrary spatial resolutions without re-training. We show how to train NeuralClothSim while imposing hard boundary conditions and demonstrate multiple applications, such as material interpolation and simulation editing. The experimental results highlight the effectiveness of our continuous neural formulation.*



## Installation

### Hardware Requirements
Clone this repository to `${code_root}`. The following sets up a conda environment with all NeuralClothSim dependencies
```
# Final
conda create --name neuralclothsim1 python=3.9
conda activate neuralclothsim1
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c bottler nvidiacub
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
or 
conda install pytorch3d -c pytorch3d


conda create --name neuralclothsim python=3.10
conda activate neuralclothsim
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c bottler nvidiacub
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
or 
conda install pytorch3d -c pytorch3d

conda create --name neuralclothsim2 python=3.9
conda activate neuralclothsim2
pip3 install --pre torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu118

conda create --name neuralclothsim python=3.9
conda activate neuralclothsim
conda install pytorch=1.13.0 torchvision cudatoolkit=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d

----------------------------------------------------------------------------------------
conda create --name neuralclothsim1 python=3.10
conda activate neuralclothsim1
conda install pytorch torchvision pytorch-cuda -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
----------------------------------------------------------------------------------------
conda install pytorch torchvision pytorch-cuda pytorch3d -c pytorch3d -c pytorch -c nvidia
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia #Fai
conda install pytorch=2.0.1 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia #Failed
conda install pytorch torchvision
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
conda install pytorch tqdm tensorboard

# Uselesss
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

or simply create from the environment.yml file
```
conda env create -f environment.yml
conda activate neuralclothsim
```

## Usage

### Running
For full version, check out train.py, reproduce simulations with
```
python train.py -c config/drape.ini -n drape_nl_canvas -m material/canvas.ini
python train.py -c config/drape.ini -n drape_linear -m material/linear_1.ini
python train.py -c config/napkin.ini --expt_name napkin -m material/canvas.ini
python train.py -c config/napkin_mesh.ini --expt_name napkin_mesh
python train.py -c config/flag_mesh.ini --expt_name flag_mesh_vis_tangents
python train.py -c config/sleeve_buckle.ini -n sleeve_buckle_canvas -m material/canvas.ini
python train.py -c config/skirt_twist.ini -n skirt_twist -m material/linear_1.ini
python train.py -c config/skirt_static_rim.ini -n skirt_static_rim -m material/canvas.ini
python train.py -c config/collision.ini -n collision_linear -m material/linear_1.ini
tensorboard --logdir ? --port=?
```

Command Line Arguments for train.py

Similarly, replacing `napkin` with `sleeve` or `skirt` will reproduce the corresponding simulations.

### Evaluation

## Citation

If you use this code for your research, please cite:
```
@article{kair2023neuralclothsim, 
	title={NeuralClothSim: Neural Deformation Fields Meet the Thin Shell Theory}, 
	author={Navami Kairanda and Marc Habermann and Christian Theobalt and Vladislav Golyanik}, 
	journal = {arXiv:2308.12970v2}, 
	year={2023} 
}
```
