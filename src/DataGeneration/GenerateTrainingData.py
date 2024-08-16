#version that outputs force as vector 3 with no quaternion
import shutil
import open3d as o3d
import numpy as np
import pickle
import os
import re



import generate_scene

from convert.voxelize import create_numpy_voxel_grid


def generate_random_force(max_magnitude=0.5):
    # Generate random angles for spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi)  # azimuthal angle
    phi = np.random.uniform(0, np.pi)        # polar angle

    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Create a unit vector
    vector = np.array([x, y, z])

    # Set new magnitude
    scale = np.random.uniform(0, 1) * max_magnitude
    vector = vector * scale

    return vector








def extract_numbers_from_file(filename, getlast=True):
    # Read the content of the file
    with open(filename, 'r') as file:
        file_content = file.read()

    # Pattern to extract numbers after "X="
    pattern = r"X=\s*([\d.eE+\-\s]+)"  # Escape the dash properly

    # Find all matches
    matches = re.findall(pattern, file_content)

    # Get the last or first match based on the getlast parameter
    if matches:
        part = matches[-1] if getlast else matches[0]
        # Split the string into individual numbers
        numbers = [float(num) for num in part.split()]
        return numbers
    else:
        return None

    
def write_to_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)



def load_obj(file_path):
    vertices = []
    faces = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.split()
                face = []
                for part in parts[1:]:
                    face.append(int(part.split('/')[0]) - 1)  # OBJ format uses 1-based indexing
                faces.append(face)
    
    return vertices, faces

def import_and_combine_meshes(folder_path, base_name):
    # List all files in the given folder
    all_files = os.listdir(folder_path)
    
    # Filter files to get those starting with base_name and ending with .obj
    obj_files = [f for f in all_files if f.startswith(base_name) and f.endswith('.obj')]
    
    if not obj_files:
        return None
    
    combined_vertices = []
    combined_faces = []
    vertex_offset = 0
    
    for obj_file in obj_files:
        file_path = os.path.join(folder_path, obj_file)
        vertices, faces = load_obj(file_path)
        
        # Add the vertices to the combined list
        combined_vertices.extend(vertices)
        
        # Adjust the face indices and add to the combined list
        for face in faces:
            adjusted_face = [index + vertex_offset for index in face]
            combined_faces.append(adjusted_face)
        
        # Update the vertex offset
        vertex_offset += len(vertices)
    
    # Create an Open3D mesh
    combined_mesh = o3d.geometry.TriangleMesh()
    combined_mesh.vertices = o3d.utility.Vector3dVector(combined_vertices)
    combined_mesh.triangles = o3d.utility.Vector3iVector(combined_faces)
    
    return combined_mesh

def remove_mtl_files(folder_path):
    # List all files in the given folder
    all_files = os.listdir(folder_path)
    
    # Filter files to get those ending with .mtl
    mtl_files = [f for f in all_files if f.endswith('.mtl')]
    
    # Remove each .mtl file
    for mtl_file in mtl_files:
        file_path = os.path.join(folder_path, mtl_file)
        os.remove(file_path)



def create_scene_instance(index, directory, mesh_dir, voxel_size, origin, boxlength):
    """
    Create a scene instance and write the force vector to a pickle file in the specified directory.

    it then runs the scene with sofa in CLI and reads the output .log file in pickle file

    Args:
        index (int): The index used to name the pickle file.
        directory (str): The directory where the pickle file will be saved.
    """
    force = generate_random_force()
    

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)



    #write simulation file
    sofa_dir = os.path.join(directory, "sofa_files")
    # Ensure the directory exists
    os.makedirs(sofa_dir, exist_ok=True)

    scenename = f"beam_scene_{index}.scn"

    sofa_file = os.path.join(sofa_dir, scenename)
    print(sofa_file)

    generate_scene.generate_file(sofa_file, force, mesh_dir)


    
    #Now we must run sim here
    # runSofa --start -n 2 --gui batch 
    # Command to execute
    command = f"cd {sofa_dir}; runSofa --start -n 2 --gui batch {scenename}"

    # Execute the command
    exit_code = os.system(command)
    # Check the exit code to determine if there were any errors
    if exit_code != 0:
        print(f"Error executing command: {command}")
    else:
        print(f"Command executed successfully: {command}")

    sofa_logfile = os.path.join(sofa_dir, f"beam_scene_{index}.log")

    output_beam = extract_numbers_from_file(sofa_logfile)

    input_beam = extract_numbers_from_file(sofa_logfile, getlast=False)#will get the undeformed beam before simulation


    #read_generated obj
    obj_dir = os.path.join(sofa_dir, "obj")
    remove_mtl_files(obj_dir)
    #import files that start with the scene name (without extention)
    combined_deformed_mesh = import_and_combine_meshes(obj_dir, os.path.splitext(scenename)[0])
 
    deformed_vertices = np.asarray(combined_deformed_mesh.vertices)
    deformed_triangles = np.asarray(combined_deformed_mesh.triangles)
    deformed_vertex_normals = np.asarray(combined_deformed_mesh.vertex_normals)

    with open("vertexCount", "a") as f:
        f.write(f"{len(combined_deformed_mesh.vertices)}\n")


    #voxelize mesh
    deformed_voxels_np = create_numpy_voxel_grid(mesh=combined_deformed_mesh, origin=origin, boxlength=boxlength, voxelsize=voxel_size)

    




    #write to pickle
    data_to_pickle = {
        'force_vector': force,
        'output_beam': output_beam,
        'input_beam' : input_beam,
        'deformed_vertices' : deformed_vertices,
        'deformed_triangles' : deformed_triangles,
        'deformed_vertex_normals' : deformed_vertex_normals,
        'deformed_voxels_np': deformed_voxels_np # the voxelized mesh as a 3d numpy array

    
    }

    pickle_dir = os.path.join(directory, "pickle_files")
    # Ensure the directory exists
    os.makedirs(pickle_dir, exist_ok=True)

    
    
    # Define the full path for the pickle file
    pickle_filename = os.path.join(pickle_dir, f"scene_data{index}.pkl")
    
    # Write the force vector to the pickle file
    write_to_pickle(pickle_filename, data_to_pickle)



def create_folder_if_not_exists(path):
    try:
        os.makedirs(path, exist_ok=True)
        #print(f"Directory '{path}' created successfully or already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

def delete_all_in_directory(directory_path):
    # Loop through the directory
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
            
if __name__ == "__main__":
    #generate and run a number of simulations

    voxel_size = 0.005  # Define an appropriate voxel size
    origin = [-.5, -.5, -.1]
    boxlength = 1 # the size of the bounding box(cube) from origin in each direction

    mesh_dir = "/media/volume/Training_data/git_repo/3dmesh_scripts/3D_models/transformed_models_decimated"

    simNumb = 400

    dir = r"/media/volume/Training_data/experiments/experiment_09/testing_data"


    with open("vertexCount", "w") as f:
        pass


    create_folder_if_not_exists(dir)
    delete_all_in_directory(dir)


    for i in range(simNumb):
        create_scene_instance(i, directory=dir, mesh_dir=mesh_dir, voxel_size=voxel_size, origin=origin, boxlength=boxlength)
    
