# Physics-based Deep Learning for Spine CT Registration
This repo presents the main results of my research internship at the Surgical Planning Lab.

<p align="center">
<img src="assets/images/simulation.gif" width=50%>
</p>

# Usage
In order to use the content of this repo you must install the required dependencies.
- You can install SOFA from: [https://www.sofa-framework.org/download/](https://www.sofa-framework.org/download/)
- Create a python virtual environement and install the requirements with pip: ```pip install -r requirements.txt```

# Data generation:
My training data is a 3D mesh from a spine CT deformed using a random set of forces.
The simulation is done using SOFA Framework.
I generate displacement fields from SOFA and run them using this python script

```src/DataGeneration/GenerateTrainingData2.py```
at the end of script you can change these parameters
```py
voxel_size = 0.005  # Define an appropriate voxel size
origin = [-.5, -.5, -.1] # origin of voxel box (from corner)
boxlength = 1 # the size of the bounding box(cube) from origin in each direction

mesh_dir = "assets/3D_models/transformed_models_decimated/"
# number of simulations to do
simNumb = 400

# the directory to save the data
dir = r"/.../testing_data" # or r"/.../training_data"

```



# Deep Learning training:
For the DL training I used pytorch and train on a voxelized version of the mesh using a feed-forward CNN regressor.
I tried directly training on the mesh using GNN but the results were not satisfying. 

# Results:

## Visual result
<p align="center">
<img src="assets/images/simulation.gif" width=50%>
</p>

On this animation we can see in green the spine simulated with the ground truth force.
The red one is the same spine but simulated using the predicted force.
As you can see the end result is visualy quite close.

## Numeric results

Validation Loss avg: 0.0072
Validation Loss Standard Deviation: 0.0118
