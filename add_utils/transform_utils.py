from spatialmath import SE3, SO3
import numpy as np
from scipy.spatial.transform import Rotation
from spatialmath.base import q2r, r2q, qisunit, qunit

class Transform_utils:
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

    def tcp2eef(self, pos, quat):
        T_Base_TCP = SE3.Rt(q2r(qunit(quat), order="xyzs"), pos)
        T_TCP_EEF = SE3.Tx(-0.15175)

        T_Base_EEF = T_Base_TCP * T_TCP_EEF

        position = T_Base_EEF.t
        orientation = r2q(T_Base_EEF.R, order="xyzs")
        return [position[0], position[1], position[2], orientation[0], orientation[1], orientation[2], orientation[3]]


    