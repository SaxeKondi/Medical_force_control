# from simulation.simulator import MJ 
# import numpy as np
# from scipy.spatial.transform import Rotation

# class Data_Query:
#     def __init__(self, rawData):
#         self.rawData = rawData

        
#     def getFTData(self):
#         return self.rawData.sensordata
    

#     def directionToNormal(self, TCP_R, _force):
#         """
#             Calulates the direction the robot should turn to align with the surface normal
#             Returns: Euler angles for rotation
#             If the end effector is parallel to the surface, the rotation matrix should be close to the identity matrix.
#         """
#         force = self.getFTData()
#         force = _force
#         force_norm = force / np.linalg.norm(force) # Normalize the force vector to be unit
#         y_axis = np.atleast_2d([0, 1, 0]) # Axis to align with
#         rot = Rotation.align_vectors(y_axis, [force_norm])[0] # Align force to y axis
#         return rot.as_matrix() @ TCP_R # New rotation matrix the robot should have to be aligned.
    
