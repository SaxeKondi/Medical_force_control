import numpy as np
from scipy.spatial.transform import Rotation
import mujoco as mj


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

    def align_with_surface_normal(self, eef_rot_matrix, surface_normal):
        """
        Aligns the end effector's x-axis with the surface normal.

        Parameters:
            eef_rot_matrix (numpy.array): The current rotation matrix of the end effector.
            surface_normal (numpy.array): The surface normal vector.

        Returns:
            numpy.array: The new orientation rotation matrix where the end effector's x-axis aligns with the surface normal.
        """
        # Compute rotation axis
        rotation_axis = np.cross(eef_rot_matrix[0], surface_normal)
        rotation_axis /= np.linalg.norm(rotation_axis)

        # Compute rotation angle
        dot_product = np.dot(eef_rot_matrix[0], surface_normal)
        rotation_angle = np.arccos(dot_product)

        # Construct rotation matrix
        rotation_matrix = Rotation.from_rotvec(rotation_angle * rotation_axis).as_matrix()

        # Apply rotation matrix
        return np.dot(rotation_matrix, eef_rot_matrix)


    def _get_sensor_force(self, site="eef_site") -> np.ndarray:
        """
        Get the sensor force at the specified site.

        Parameters:
            site (str): The name of the site.

        Returns:
            np.ndarray: The sensor force.
        """
        eef_rot_mat = self.data.site_xmat[self.model_names.site_name2id[site]].reshape(3, 3)
        force = self.data.sensordata[:3]  # only forces
        return eef_rot_mat @ force, eef_rot_mat # Returns force in world frame


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
        is_in_contact, cs_i = self._is_in_contact(obj1, obj2)

        if is_in_contact:
            wrench = self._get_cs(cs_i)
            contact_frame = self.data.contact[cs_i].frame.reshape((3, 3)).T
            return contact_frame @ wrench[:3]
        else:
            return np.zeros(3, dtype=np.float64)


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
