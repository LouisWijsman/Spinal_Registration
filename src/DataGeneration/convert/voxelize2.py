import numpy as np
import open3d as o3d


def voxel_grid_to_numpy(voxel_grid, target_shape):
    # Compute the voxel grid dimensions
    
    voxel_size = voxel_grid.voxel_size
    min_bound = voxel_grid.get_min_bound()
    max_bound = voxel_grid.get_max_bound()
    grid_dimensions = np.ceil((max_bound - min_bound) / voxel_size).astype(int)

    # Create an empty grid
    grid = np.zeros(grid_dimensions, dtype=np.float32)

    # Iterate over all voxels in the voxel grid
    for voxel in voxel_grid.get_voxels():
        x, y, z = voxel.grid_index
        grid[x, y, z] = 1.0  # Mark the voxel as occupied

    # Pad or crop the grid to match the target shape
    padded_grid = np.zeros(target_shape, dtype=np.float32)
    min_dims = np.minimum(grid_dimensions, target_shape)
    padded_grid[:min_dims[0], :min_dims[1], :min_dims[2]] = grid[:min_dims[0], :min_dims[1], :min_dims[2]]

    return padded_grid, voxel_size





def create_voxel_grid(mesh, voxel_size):    
    # Create a VoxelGrid from the TriangleMesh within the specified bounds
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(input=mesh, voxel_size=voxel_size)
    return voxel_grid


def create_numpy_voxel_grid(mesh, voxelsize):
    """make a numpy grid from a mesh
    origin: is the 3d coordinates in space of the origin
    boxlength: is the length in each direction from origin to form voxel box
    voxelsize: is the size of a voxel

    the output is a 3d cube numpy array with as many voxels as fit in the box of boxlength
    """
    #convert mesh to voxels
    voxels = create_voxel_grid(mesh, voxelsize)

    voxels_np, vox_size = voxel_grid_to_numpy(voxels, [60, 60, 60])

    return voxels_np