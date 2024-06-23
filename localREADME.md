# NeuralClothSim: Neural Deformation Fields Meet the Thin Shell Theory


## Installation

The following sets up a conda environment with all NeuralClothSim dependencies

```
conda env create -f environment.yml
conda activate neuralclothsim
```

## Usage
For simpler version, check out neural_cloth_sim.py

For full version, check out train.py, reproduce simulations with
```
python train.py -c config/drape.ini -n drape_nl_canvas -m material/canvas.ini
python train.py -c config/napkin.ini --expt_name napkin -m material/canvas.ini
python train.py -c config/napkin_mesh.ini --expt_name napkin_mesh
python train.py -c config/flag_mesh.ini --expt_name flag_mesh_vis_tangents
python train.py -c config/sleeve_buckle.ini -n sleeve_buckle_canvas -m material/canvas.ini
python train.py -c config/skirt_twist.ini -n skirt_twist -m material/linear_1.ini
python train.py -c config/skirt_static_rim.ini -n skirt_static_rim -m material/canvas.ini
python train.py -c config/collision.ini -n collision_linear -m material/linear_1.ini
tensorboard --logdir ? --port=?
```

Similarly, replacing `napkin` with `sleeve` or `skirt` will reproduce the corresponding simulations.
