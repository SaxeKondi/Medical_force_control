import numpy as np
from colect_sim.env.ur5_env import UR5Env
from colect_sim.utils.traj_generation import linear_traj_w_gauss_noise
from spatialmath.base import q2r, r2q

import trimesh
import os

# Load the STL file
mesh = trimesh.load_mesh(os.path.dirname(os.path.realpath(__file__)) + '/scene/meshes/phantom/belly.stl')

def sample_point_on_mesh(mesh, x, y):
    # Define a ray from the given (x, y) position towards the mesh
    direction = [0, 0, -1]  # Assuming the ray direction is straight down
    origin = [x, y, 1000]  # Assuming a high enough starting z value

    # Ensure the vertices are in the correct shape
    vertices = mesh.vertices.reshape((-1, 3))

    # Ensure the triangles are in the correct shape
    triangles = mesh.faces.reshape((-1, 3))

    # Create a trimesh object
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    # Perform ray-mesh intersection
    intersection_info = trimesh_mesh.ray.intersects_location([origin], [direction])

    if len(intersection_info[0]) > 0:
        # If intersection is found, return the intersection point
        intersection_point = intersection_info[0][1]
        return intersection_point + [0.0, 0.5, 0.0]

    
def belly_traj():
    # Define the range of y-coordinates
    # y_range = np.arange(-0.21, 0.22, 0.01)
    # x_range = np.arange(-0.2, 0.2, 0.02)
    x_range = np.arange(-0.16, 0.16, 0.02)
    
    points = []

    for x in x_range:
        points.append(sample_point_on_mesh(mesh, x, 0.0))

    points = np.array(points)
    
    # Define the quaternion for all points
    quat = np.array([0, 1, 0, 1])
    quat = quat / np.linalg.norm(quat)

    # Repeat the quaternion for each point
    quats = np.tile(quat, (len(points), 1))
    
    # Return the points and quaternions
    return np.hstack((points, quats))


def main() -> None:
    env = UR5Env()

    # quat = np.array([0,1,0,1]) # in z direction
    # quat = np.array([0,0,0,1]) # in x direction # rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    quat = np.array([0,1,0,0]) # in -x direction # rot = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    # quat = np.array([0.00000, 0.00000, 0.5, 1.11803]) # in x-y direction # rot = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]])
    # quat = np.array([-0.25, 0.55902, 0.55902, 1.25]) # in x-y-z direction # rot = np.array([[1, -1, 1], [1, 1, 1], [-1, 0, 1]])
    quat = quat / np.linalg.norm(quat)
    
    
    # Linear scanning near one edge
    # traj_start = np.array([0.375, 0.46, 0.205,quat[0],quat[1],quat[2],quat[3]])
    # traj_stop = np.array([0.625, 0.45, 0.205,quat[0],quat[1],quat[2],quat[3]])


    # traj_start = np.array([0.42, 0.45, 0.213,quat[0],quat[1],quat[2],quat[3]])
    # traj_stop = np.array([0.7, 0.45, 0.205,quat[0],quat[1],quat[2],quat[3]])

    # [-0.19183677 -0.04010982  0.05729382]
    # 0.52519069 0.40056881 0.23105277
    # traj_start = np.array([0.52519069, 0.40056881, 0.23105277+0.15175,quat[0],quat[1],quat[2],quat[3]])
    # traj_stop = np.array([0.30816323, 0.45989018, 0.05729382+0.15175,quat[0],quat[1],quat[2],quat[3]])

    #############
    # For testing the correct target force for the controller 

    # Move in z direction
    # traj_start = np.array([0.5, -0.25, 0.3,quat[0],quat[1],quat[2],quat[3]])
    # traj_stop = np.array([0.5, 0.25, 0.3,quat[0],quat[1],quat[2],quat[3]])  

    # Move in x direction
    # traj_start = np.array([0.5, 0.25, 0.625,quat[0],quat[1],quat[2],quat[3]])
    # traj_stop = np.array([0.5, 0.25, 0.15175,quat[0],quat[1],quat[2],quat[3]])

    # Move in -x direction
    traj_start = np.array([-0.5, -0.25, 0.625,quat[0],quat[1],quat[2],quat[3]])
    traj_stop = np.array([-0.5, -0.25, 0.15175,quat[0],quat[1],quat[2],quat[3]])  
    
    #############

    # Move in pos x direction
    # traj_start = np.array([0.35, 0.45, 0.21 - 0.15175,quat[0],quat[1],quat[2],quat[3]])
    # traj_stop = np.array([0.5, 0.45, 0.21 - 0.15175,quat[0],quat[1],quat[2],quat[3]])

    # Move in neg x direction
    # traj_start = np.array([0.68, 0.45, 0.21,quat[0],quat[1],quat[2],quat[3]])
    # traj_stop = np.array([0.5, 0.45, 0.21,quat[0],quat[1],quat[2],quat[3]])

    # Move in pos y direction
    # traj_start = np.array([0.5, 0.35, 0.21,quat[0],quat[1],quat[2],quat[3]])
    # traj_stop = np.array([0.5, 0.45, 0.21,quat[0],quat[1],quat[2],quat[3]])

    traj = linear_traj_w_gauss_noise(traj_start, traj_stop, 100, 0., 0.0005)
    # traj = belly_traj()
    # traj = traj[::-1] # Reverse the array
    # mid_traj_pose = traj[len(traj)//2 - 7]
    # traj = traj[len(traj)//2 - 7:]
    # # print(traj)
    # traj = traj[::2]

    # traj = np.array([mid_traj_pose + [0, 0, 0.1, 0, 0, 0, 0], mid_traj_pose, traj[-2]])
    # # traj = np.insert(traj, 0, mid_traj_pose + [0, 0, 0.1, 0, 0, 0, 0], axis=0)
    # # print(traj)


    i = 0
    terminated = False
    while not terminated:
        next = traj[i]
        op_target_reached = False
        while not op_target_reached:
            op_target_reached, terminated = env.step(next, controller_name="admittance", i = i) # Controller options: "op_space" or "admittance"
        env.enable_recording = False # inelegant, but works for aligning the recording to the target
        i += 1
        if i > len(traj) - 1 : terminated = True


if __name__ == "__main__":
    main()