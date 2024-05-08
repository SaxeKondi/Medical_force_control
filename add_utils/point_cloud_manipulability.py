import numpy as np
import os
import open3d as o3d


######################################################################################################
# Class for working with point clouds.
# Includes the following functions
#       - create_point_cloud
#       - update_point_cloud
#       - load_from_object_file
#       - save_point_cloud
#       - load_point_cloud
#       - get_dimensions
#       - create_train_data
#       - min_max_rounded_cube
#       - generate_Xstar
######################################################################################################
class Point_cloud_Manipulability:
    def __init__(self):
        """
        Initializes a Point_cloud object.
        """
        self.point_cloud = o3d.geometry.PointCloud()


    def load_from_object_file(self, stl_file_name = "Belly.stl", obj_translate = [0.0, 0.0, 0.0], scale_factor = 0.6666, num_points = 200):
        """
        Loads a point cloud from an STL file.

        Parameters
        ----------
            stl_file_name (str): Name of the STL file.
            num_points (int): Number of points to sample from the mesh.
        """
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

        # Convert points to NumPy array for scaling
        points_array = np.asarray(self.point_cloud.points)

        # Scale the point cloud along the y-axis
        points_array[:, 0] *= scale_factor

        # Update point cloud with scaled points
        self.point_cloud.points = o3d.utility.Vector3dVector(points_array)

        # Estimate surface normals pointing away from the object surface (not inside)
        self.point_cloud.estimate_normals()


    def sample_points_above_z(self, z_threshold):
        """
        Sample points from the point cloud where the z-coordinate is above a threshold.

        Parameters
        ----------
        z_threshold : float
            Threshold value for the z-coordinate.

        Returns
        -------
        open3d.geometry.PointCloud
            Point cloud containing only points where the z-coordinate is above the threshold.
        """
        # Convert point cloud to NumPy array
        points_array = np.asarray(self.point_cloud.points)
        normals_array = np.asarray(self.point_cloud.normals)

        # Filter points based on z-coordinate
        filtered_indices = points_array[:, 2] > z_threshold
        filtered_points = points_array[filtered_indices]
        filtered_normals = normals_array[filtered_indices]

        # Create a new point cloud with the filtered points and normals
        self.filtered_point_cloud = o3d.geometry.PointCloud()
        self.filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
        self.filtered_point_cloud.normals = o3d.utility.Vector3dVector(filtered_normals)



    def visu_point_cloud(self):
        #################
        # Visualization #
        #################
        # Create axes
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        # Visualize the point cloud, axes, and point of interest sphere
        o3d.visualization.draw_geometries([self.point_cloud, axes])


if __name__ == "__main__":
    point_cloud = Point_cloud_Manipulability()

    point_cloud.load_from_object_file(stl_file_name="Belly_old.stl", obj_translate=[0.5, 0.5, 0.0], scale_factor=0.6666, num_points=200)

    # Sample points where z-coordinate is above 0
    point_cloud.sample_points_above_z(z_threshold=0.001)

    point_cloud.visu_point_cloud()
