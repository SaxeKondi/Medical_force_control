#Admittance Controller

import numpy as np
from scipy.spatial.transform import Rotation


def admittance(target_force, wrench, des_vel, des_acc, des_pos, rot_align, current_pos, current_vel):

    '''
    INPUTS:
        target_force -> force to push against object (surface normal)
        wrench -> the forces and torques exerted by the surface on the end effector due to contact. (6 element vector, [Fx; Fy; Fz, τx; τy; τz;])
        des_vel -> desired velocity during movement. Constant
        des_acc -> desired acceleration during movement. Constant
        des_pos -> where to map next, computed by rule
        rot_align -> rotation matrix the robot should have to be aligned with surface normal
        current_pos 
        current_vel
    
    RETUNS:
    Xe = desired end effector position based on controller response to external force (surface normal) in TOOL FRAME
    '''

    m1 = 1
    m2 = 1
    k1 = 1
    k2 = 1
    kenv1 = 10 #set by user depending on current object to be reconstructed
    kenv2 = 10
    kd1 = 2*np.sqrt(m1*(k1+kenv1))
    kd2 = 2*np.sqrt(m2*(k2+kenv2))

    dt = 0.01 #PLACEHOLDER (we need to think what is dt for only one step)

    M_prev = [
        [m1,0,0],[0,m2,0],[0,0,0]
    ]

    K_prev = [
        [k1,0,0],[0,k2,0],[0,0,0]
    ]
    
    D_prev = [[kd1,0,0],[0,kd2,0],[0,0,0]]

    M = rot_align @ M_prev #update gains based on orientation function
    K = rot_align @ K_prev
    D = rot_align @ D_prev
    
    # Calculate acceleration using 
    acc = np.linalg.inv(M) * (target_force - wrench - K @ current_pos - D @ current_vel)

    # Integrate acceleration to get velocity
    vel = des_vel + acc * dt  # Euler's method (delta v = a * delta t)

    # Integrate velocity to get position
    Xe = des_pos + vel * dt  # Euler's method (delta x = v * delta t)


    Xc = Xe + des_pos 
    return Xc

def next_step():
    des_pos = []

    #this function computes the desired next step based on rule, such as move small step along surface (same orientation)

    return des_pos

def directionToNormal(TCP_R, force):

    """
        Inputs: TCP rotation (3x3 matrix), force direction (3x1 vector XYZ)
        Calulates the direction the robot should turn to align with the surface normal
        Returns: Euler angles for rotation
        If the end effector is parallel to the surface, the rotation matrix should be close to the identity matrix.
    """
    force_norm = force / np.linalg.norm(force) # Normalize the force vector to be unit
    z_axis = np.atleast_2d([0, 0, 1]) # Axis to align with
    rot = Rotation.align_vectors(z_axis, [force_norm])[0] # Align force to z axis
    return rot.as_matrix() @ TCP_R # New rotation matrix the robot should have to be aligned.

def desired_force():
    Kp = 10 #set by user based on stiffness

    force_value = 2 + Kp

    return force_value

def tool_to_base(tool_frame):

    """
    Transform a 4x4 T matrix in tool_frame to base frame.
    """

    T_base_tool = np.array([
    [-0.52960, 0.74368, 0.40801, 0.27667],
    [0.84753, 0.44413, 0.29059, -0.60033],
    [0.03490, 0.49970, -0.86550, 0.51277],
    [0.00000, 0.00000, 0.00000, 1.00000]
    ])

    T_tool_tcp = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0.00000, 0.00000, 0.00000, 1.00000]])
    
    # Multiply tool_frame by the identity matrix
    final = tool_frame @ T_base_tool @ T_tool_tcp

    return final