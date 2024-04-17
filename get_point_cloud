import open3d as o3d
import numpy as np

# Load OBJ file
mesh = o3d.io.read_triangle_mesh("scene/meshes/phantom/phantom.obj")

# Convert to point cloud
point_cloud = mesh.sample_points_uniformly(number_of_points=10000)

point_cloud.estimate_normals()

# print(np.asarray(point_cloud.normals))

print(np.asarray(point_cloud.points))

# Visualize the point cloud
# o3d.visualization.draw_geometries([point_cloud])


#################################
# New point with unknown normal #
#################################
kd_tree = o3d.geometry.KDTreeFlann(point_cloud)

# Point of interest (replace with your desired point)
point_of_interest = np.array([0.0, 0.0, 0.0])

# Define search parameters (adjust radius as needed)
search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)

# Estimate normals at the point of interest
[k, indices, _] = kd_tree.search_radius_vector_3d(point_of_interest, 0.01)  # You can adjust the radius as needed

normal_at_point = np.asarray(point_cloud.normals)[indices[0], :]

print("Normal at point of interest:", normal_at_point)



############################
# Get dimensions of object #
############################

# Compute the axis-aligned bounding box
axis_aligned_bounding_box = point_cloud.get_axis_aligned_bounding_box()

# Get the dimensions of the bounding box
dimensions = axis_aligned_bounding_box.get_extent()
print("Dimensions of the object (width, height, depth):", dimensions)


#################
# Visualization #
#################
visu = False
if visu:
    # Visualization
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()

    # Add point cloud
    visualizer.add_geometry(point_cloud)

    # Add axis-aligned bounding box
    visualizer.add_geometry(axis_aligned_bounding_box)

    # Set the view control
    view_control = visualizer.get_view_control()
    view_control.rotate(180.0, 0.0)
    view_control.translate(0.0, 0.0)

    visualizer.run()
    visualizer.destroy_window()