import numpy as np
import trimesh

# Load the STL file
mesh = trimesh.load_mesh('C:/Users/fanoj/OneDrive - Syddansk Universitet/Master/2. Semester/Introduction to Medical Robotics/Project/Medical_force_control/scene/meshes/phantom/belly.stl')

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
        return intersection_point + [0.5, 0.5, 0.0]
    else:
        # If no intersection is found, return None
        return None
    

def belly_traj():
    # Define the range of y-coordinates
    y_range = np.arange(0.0, 0.1, 0.02)
        
    points = []

    for y in y_range:
        points.append(sample_point_on_mesh(mesh, y, 0.0))

    points = np.array(points)
    return points


print(belly_traj())

# print(sample_point_on_mesh(mesh, 0.1, 0.0))