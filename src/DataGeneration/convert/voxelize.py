import numpy as np
import open3d as o3d


def voxel_grid_to_numpy(voxel_grid, grid_dimensions):
    # Compute the voxel grid dimensions
    

    # Create an empty grid of the desired size
    voxels_np = np.zeros(grid_dimensions, dtype=np.float32)

    # Iterate over all voxels in the voxel grid to populate the empty grid
    for voxel in voxel_grid.get_voxels():
        x, y, z = voxel.grid_index
        voxels_np[x, y, z] = 1.0  # Mark the voxel as occupied

    return voxels_np




def create_voxel_grid(mesh, voxel_size, min_bound_origin):    
    # Create a VoxelGrid from the TriangleMesh within the specified bounds
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(input=mesh, voxel_size=voxel_size, min_bound=min_bound_origin, max_bound=[0, 0, 0])
    return voxel_grid


def create_numpy_voxel_grid(mesh, origin, boxlength, voxelsize):
    """make a numpy grid from a mesh
    origin: is the 3d coordinates in space of the origin
    boxlength: is the length in each direction from origin to form voxel box
    voxelsize: is the size of a voxel

    the output is a 3d cube numpy array with as many voxels as fit in the box of boxlength
    """
    grid_size = int(boxlength / voxelsize)
    #convert mesh to voxels
    voxels = create_voxel_grid(mesh, voxelsize, origin)

    voxels_np = voxel_grid_to_numpy(voxels, [grid_size, grid_size, grid_size])

    return voxels_np