#generate the scene file automaticaly as an xml file
import os

#this list will contain the finished file piece by piece
file = []

#first part of the file
file.append("""
<?xml version="1.0"?>

<Node name="root" dt="0.01" gravity="0 0 0">
    <RequiredPlugin name="Sofa.Component.LinearSolver.Direct"/>             <!-- Needed to use components [BTDLinearSolver] -->
    <RequiredPlugin name="Sofa.Component.Mapping.Linear"/>                  <!-- Needed to use components [BeamLinearMapping IdentityMapping] -->
    <RequiredPlugin name="Sofa.Component.Mass"/>                            <!-- Needed to use components [UniformMass] -->
    <RequiredPlugin name="Sofa.Component.ODESolver.Backward"/>              <!-- Needed to use components [EulerImplicitSolver] -->
    <RequiredPlugin name="Sofa.Component.SolidMechanics.FEM.Elastic"/>      <!-- Needed to use components [BeamFEMForceField] -->
    <RequiredPlugin name="Sofa.Component.StateContainer"/>                  <!-- Needed to use components [MechanicalObject] -->
    <RequiredPlugin name="Sofa.Component.AnimationLoop"/>                   <!-- Needed to use components [FreeMotionAnimationLoop] -->
    <RequiredPlugin name="Sofa.Component.Constraint.Projective"/>           <!-- Needed to use components [FixedConstraint] -->
    <RequiredPlugin name="Sofa.Component.Constraint.Lagrangian.Correction"/><!-- Needed to use components [LinearSolverConstraintCorrection] -->
    <RequiredPlugin name="Sofa.Component.Constraint.Lagrangian.Solver"/>    <!-- Needed to use components [GenericConstraintSolver] -->
    <RequiredPlugin name="Sofa.Component.MechanicalLoad"/>                  <!-- Needed to use components [ConstantForceField] -->
    <RequiredPlugin name="Sofa.Component.Topology.Container.Constant"/>     <!-- Needed to use components [MeshTopology] -->
    <RequiredPlugin name="Sofa.Component.Visual"/>                          <!-- Needed to use components [VisualStyle] -->
    
    <DefaultAnimationLoop/>
    <VisualStyle displayFlags="showBehaviorModels showForceFields showCollisionModels showVisual showInteractionForceFields" />

    <Node name="beam">
        <StaticSolver newton_iterations="10"/>
        <BTDLinearSolver template="BTDMatrix6d" printLog="false" verbose="false" />
        <MechanicalObject template="Rigid3" name="DOFs" 
            position="
""")
#[1]placeholder
file.append("coordinates") 

#append part after coordinates
file.append("""
            "
            showIndices="false" showIndicesScale="0"/>
        <MeshTopology name="lines" lines="0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9" />
        <FixedConstraint name="FixedConstraint" indices="0" />
        <UniformMass totalMass="0.18" printLog="false" showAxisSizeFactor="0.0" />
        <BeamFEMForceField name="FEM" radius="0.01" radiusInner="0" youngModulus="2000000" poissonRatio="0.45"/>
            """)
#[3]placeholder
file.append("set_force and writestate")

#[4]placeholder
file.append("generate_visual_spine_lines")

file.append("""
        


    </Node>
</Node>
""")


def set_force(vector, writestate_file="transform_log.txt"):
    #here we set the force to apply to the end of the spine as well as the file to write out the simulation data
    lines = []
    lines.append(f'        <ConstantForceField indices="9" totalForce="{vector[0]} {vector[1]} {vector[2]} 0 0 0 1" showArrowSize="0.15"/>')

    lines.append(f'        <WriteState filename="{writestate_file}" WriteX="true"/>')
    lines.append("")

    return "\n".join(lines)


def generate_visual_spine_lines(n, filename, directory):
    """generate the lines for mapping 3d mesh"""
    lines = []
    for i in range(n):
        lines.append(f'        <Node name="Visual_spine" gravity="0 0 0">')
        lines.append(f'            <MeshOBJLoader name="ObjLoader{i}" filename="{os.path.join(directory, f"transformed_Segment_{i+15}.obj")}" scale3d="1 1 1" translation="0 0 0" rotation="0 0 0"/>')
        lines.append(f'            <OglModel name="Visual{i}" src="@ObjLoader{i}" />')
        lines.append(f'            <RigidMapping winput="@DOFs" output="@Visual{i}"  index="{i}"/>')
        lines.append(f'            <ObjExporter filename="{filename}_simulated_segment_{i}" edges="0" triangles="1" tetras="0" listening="true" exportAtEnd="true"/>')
        lines.append(f'        </Node>')

        lines.append("")
    return "\n".join(lines)


def read_file_to_string(filename):
    with open(filename, 'r') as f:
        content = f.read()
    return content

def write_file(filename, content):
       
       with open(filename, 'w') as f:
        f.write(content)
        
def generate_file(filename, force, mesh_dir):
    """main function that does generate the sofa .scn file"""
    #get filename
    logfilename = os.path.basename(filename)
    #remove extention
    logfilename = os.path.splitext(logfilename)[0]
    file[3] = set_force(force, f"{logfilename}.log")

    #get path for obj file to export from sim
    obj_name = os.path.basename(filename)
    obj_name = os.path.splitext(obj_name)[0]
    #put everyting inside obj folder
    obj_name = os.path.join("obj", obj_name)

    file[4] = generate_visual_spine_lines(10, obj_name, directory=mesh_dir)


    file[1] = read_file_to_string("src/DataGeneration/spine_nodes_position_rotation.txt")
    #join everything creating new lines wen needed
    content = "\n".join(file)
    write_file(filename, content)






if __name__ == "__main__":
    generate_file("test_scene_generated.scn", (0, 1, 0), "/media/volume/Training_data/git_repo/3dmesh_scripts/3D_models/transformed_models_decimated")
    print("New scene generatred!")