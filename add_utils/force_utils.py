import numpy as np
from scipy.spatial.transform import Rotation
import mujoco as mj
from spatialmath import SE3, SO3


class Force_utils:
    """
    A class containing utility functions related to force calculations.
    """

    def __init__(self, model, data, model_names):
        """
        Initialize the Force_utils object.

        Parameters:
            model: The Mujoco model.
            data: The Mujoco data associated with the model.
            model_names: Object containing model names mappings.
        """
        self.model = model
        self.data = data
        self.model_names = model_names

    def align_with_surface_normal(self, surface_normal):
        """
        Aligns the end effector's x-axis with the surface normal.

        Parameters:
            eef_rot_matrix (numpy.array): The current rotation matrix of the end effector.
            surface_normal (numpy.array): The surface normal vector.

        Returns:
            numpy.array: The new orientation rotation matrix where the end effector's x-axis aligns with the surface normal.
        """
        z_axis = np.array([-1, 0, 0])
        rotation_axis = np.cross(z_axis, surface_normal)
        rotation_angle = np.arccos(np.dot(z_axis, surface_normal))

        # Construct rotation matrix
        rotation_matrix = Rotation.from_rotvec(rotation_angle * rotation_axis).as_matrix()

        # Apply rotation matrix
        return rotation_matrix


    def _get_sensor_force(self, site="eef_site") -> np.ndarray:
        """
        Get the sensor wrench at the specified site.

        Parameters:
            site (str): The name of the site.

        Returns:
            np.ndarray: The sensor wrench.
        """
        eef_rot_mat = self.data.site_xmat[self.model_names.site_name2id[site]].reshape(3, 3)
        wrench = self.data.sensordata
        return eef_rot_mat @ wrench, eef_rot_mat # Returns wrench in world frame


    def _obj_in_contact(self, cs, obj1: str, obj2: str) -> bool:
        """
        Check if the specified objects are in contact in the given contact sensor.

        Parameters:
            cs: Contact sensor.
            obj1 (str): Name of object 1.
            obj2 (str): Name of object 2.

        Returns:
            bool: True if objects are in contact, False otherwise.
        """
        cs_ids = [cs.geom1, cs.geom2]

        if obj1 == "softbody":
            ob1_id = 34
            ob2_id = -1
        elif obj1 == "box":
            ob1_id = 34
            ob2_id = self.model.geom("prop").id
        elif obj1 == "belly":
            ob1_id = 34
            ob2_id = self.model.geom("belly").id
        obj_ids = [ob1_id, ob2_id]

        if all(elem in cs_ids for elem in obj_ids):
            return True
        else:
            return False


    def _is_in_contact(self, obj1: str = "", obj2: str = "") -> tuple[bool,int]:
        """
        Check if the specified objects are in contact.

        Parameters:
            obj1 (str): Name of object 1.
            obj2 (str): Name of object 2.

        Returns:
            tuple[bool, int]: A tuple containing a boolean indicating if the objects are in contact and the index of the contact if applicable.
        """
        i = 0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if self._obj_in_contact(contact, obj1, obj2):
                return (True, i)
        return (False, i)


    def _get_contact_info(self, obj1: str = "", obj2: str = "") -> np.ndarray:
        """
        Get the force and torque values from the simulation for the specified objects.

        Parameters:
            obj1 (str): Name of object 1.
            obj2 (str): Name of object 2.

        Returns:
            np.ndarray: Numpy array containing the force and torques.
        """
        is_in_contact, cs_i = self._is_in_contact(obj1)

        if is_in_contact:
            wrench = self._get_cs(cs_i)
            contact_frame = self.data.contact[cs_i].frame.reshape((3, 3)).T
            return contact_frame @ wrench[:3], contact_frame, True
        else:
            return np.zeros(3, dtype=np.float64), np.zeros([3, 3], dtype=np.float64), False


    def _get_cs(self, i: int) -> list[float]:
        """
        Get the contact force and torque for the specified contact index.

        Parameters:
            i (int): Index of the contact.

        Returns:
            list[float]: List containing the contact force and torque.
        """
        c_array = np.zeros(6, dtype=np.float64)
        mj.mj_contactForce(self.model, self.data, i, c_array)
        return c_array


    def get_ee_transformation(self, rot, ee_position, desired_rot):
        z_direction = rot[:, -1]
        z_offset = .15175
        tool_position = ee_position + z_direction * z_offset 
        rotated_tool_pose = SE3.Rt(desired_rot, tool_position)
        rotated_z = rotated_tool_pose.R[:, -1]
        ee_new_pos = rotated_tool_pose.t - rotated_z * z_offset
        return SE3.Rt(desired_rot, ee_new_pos)
    

    def _rotation_matrix_to_align_z_to_direction(self, direction):
        # Normalize direction vector
        direction /= np.linalg.norm(direction)
        # print("normalized force: ", direction)

        # Calculate axis of rotation
        axis = np.cross([0, 0, 1], direction)
        axis /= np.linalg.norm(axis)

        # Calculate angle of rotation
        angle = np.arccos(np.dot([0, 0, 1], direction))

        # Construct rotation matrix using axis-angle representation
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        x, y, z = axis
        rotation_matrix = np.array([[t*x*x + c, t*x*y - z*s, t*x*z + y*s],
                                    [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
                                    [t*x*z - y*s, t*y*z + x*s, t*z*z + c]])

        return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation_matrix