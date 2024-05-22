import open3d as o3d
import numpy as np
import os

# Class that creates a point cloud given the dimensions of a 3D rectangle
# Includes a function to return the surface normal given a point of interest 
# based on the closest point in the point cloud
class Point_cloud:
    def __init__(self, stl_file_name = "Belly.stl", obj_translate = [0.0, 0.5, 0.0], num_points = 10000):
        # Define dimensions
         # Get the directory path of the current Python script
        script_dir = os.path.dirname(__file__)

        # Navigate to the parent directory of the script
        parent_dir = os.path.dirname(script_dir)

        print(parent_dir)

        # Load OBJ file
        mesh = o3d.io.read_triangle_mesh(parent_dir + "/scene/meshes/phantom/" + stl_file_name)

        # Convert to point cloud
        self.point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)

        # Translate point cloud
        self.point_cloud.translate(obj_translate)

        # Estimate surface normals pointing away from the object surface (not inside)
        self.point_cloud.estimate_normals()

        # Create kd tree for nearest neighbor search
        self.kd_tree = o3d.geometry.KDTreeFlann(self.point_cloud)


    def get_surface_normal(self, tool_tip_point, print_normal=False):
        #######################################################
        # Get surface normal of new point with unknown normal #
        #######################################################
        self.tool_tip_point = tool_tip_point
        # Estimate normals at the point of interest
        [k, indices, _] = self.kd_tree.search_radius_vector_3d(tool_tip_point, 100)  # You can adjust the radius as needed

        normal_at_point = np.asarray(self.point_cloud.normals)[indices[0], :]

        if print_normal:
            print("Normal at point of interest:", normal_at_point)

        return normal_at_point

    def visu_point_cloud(self, visu_point_of_interest=False):
        #################
        # Visualization #
        #################
        if visu_point_of_interest:
            # Create axes
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

            poi_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            poi_sphere.translate(self.tool_tip_point)

            # Visualize the point cloud, axes, and point of interest sphere
            o3d.visualization.draw_geometries([self.point_cloud, axes, poi_sphere])
        else:
            # Create axes
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

            # Visualize the point cloud, axes, and point of interest sphere
            o3d.visualization.draw_geometries([self.point_cloud, axes])

        
if __name__ == "__main__":
    point_cloud = Point_cloud()

    # Point of interest (replace with your desired point)
    point_of_interest = np.array([0.42, 0.46, 1.06])

    point_cloud.get_surface_normal(point_of_interest)
    point_cloud.visu_point_cloud()