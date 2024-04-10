import numpy as np
from scipy.spatial.transform import Rotation

def directionToNormal(TCP_R, force):
    """
        Inputs: TCP rotation, force direction
        Calulates the direction the robot should turn to align with the surface normal
        Returns: Euler angles for rotation
        If the end effector is parallel to the surface, the rotation matrix should be close to the identity matrix.
    """
    force_norm = force / np.linalg.norm(force) # Normalize the force vector to be unit
    z_axis = np.atleast_2d([0, 0, 1]) # Axis to align with
    rot = Rotation.align_vectors(z_axis, [force_norm])[0] # Align force to z axis
    return rot.as_matrix() @ TCP_R # New rotation matrix the robot should have to be aligned. 