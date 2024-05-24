import numpy as np

from enum import Enum
from colect_sim.controller.joint_effort_controller import JointEffortController
from colect_sim.utils.mujoco_utils import get_site_jac, get_fullM
from colect_sim.utils.controller_utils import task_space_inertia_matrix, pose_error
from colect_sim.utils.mujoco_utils import MujocoModelNames
from mujoco import MjModel, MjData
from typing import List

from add_utils.point_cloud import Point_cloud
from add_utils.force_utils import Force_utils
from add_utils.transform_utils import Transform_utils
from scipy.spatial.transform import Rotation
from spatialmath.base import q2r, r2q
from spatialmath import SE3, SO3

import os


class TargetType(Enum):
    POSE = 0
    TWIST = 1

class OperationalSpaceController(JointEffortController):
    def __init__(
        self,
        model: MjModel,
        data: MjData,
        model_names: MujocoModelNames,
        eef_name: str,
        joint_names: List[str],
        actuator_names: List[str],
        min_effort: List[float],
        max_effort: List[float],
        target_type: TargetType,
        kp: float,
        ko: float,
        kv: float,
        vmax_xyz: float,
        vmax_abg: float,
        null_damp_kv: float,
    ) -> None:
        """
        Operational Space Controller class to control the robot's joints using operational space control with gravity compensation.

        Parameters:
            model (mujoco._structs.MjModel): Mujoco model representing the robot.
            data (mujoco._structs.MjData): Mujoco data associated with the model.
            model_names (homestri_ur5e_rl.utils.mujoco_utils.MujocoModelNames): Names of different components in the Mujoco model.
            eef_name (str): Name of the end-effector in the Mujoco model.
            joint_names (List[str]): List of joint names for the robot.
            actuator_names (List[str]): List of actuator names for the robot.
            min_effort (List[float]): List of minimum allowable effort (torque) for each joint.
            max_effort (List[float]): List of maximum allowable effort (torque) for each joint.
            target_type (TargetType): The type of target input for the controller (POSE or TWIST).
            kp (float): Proportional gain for the PD controller in position space.
            ko (float): Proportional gain for the PD controller in orientation space.
            kv (float): Velocity gain for the PD controller.
            vmax_xyz (float): Maximum velocity for linear position control.
            vmax_abg (float): Maximum velocity for orientation control.
            ctrl_dof (List[bool]): Control degrees of freedom for each joint (True if controlled, False if uncontrolled).
            null_damp_kv (float): Damping gain for null space control.
        """
        super().__init__(model, data, model_names, eef_name, joint_names, actuator_names, min_effort, max_effort)

        self.target_type = target_type
        self.kp = kp
        self.ko = ko
        self.kv = kv
        self.vmax_xyz = vmax_xyz
        self.vmax_abg = vmax_abg
        self.null_damp_kv = null_damp_kv

        self.task_space_gains = np.array([self.kp] * 3 + [self.ko] * 3)
        self.lamb = self.task_space_gains / self.kv
        self.sat_gain_xyz = vmax_xyz / self.kp * self.kv
        self.sat_gain_abg = vmax_abg / self.ko * self.kv
        self.scale_xyz = vmax_xyz / self.kp * self.kv
        self.scale_abg = vmax_abg / self.ko * self.kv

        self.actual_pose = None
        self.target_pose = None
        self.target_tol = 0.001#0.0075#0.01
        self.actual_wrench = None


    def run(self, target: np.ndarray, ctrl: np.ndarray) -> None:
        """
        Run the operational space controller to control the robot's joints using operational space control with gravity compensation.

        Parameters:
            target (numpy.ndarray): The desired target input for the controller.
            ctrl (numpy.ndarray): Control signals for the robot actuators.

        Notes:
            The controller sets the control signals (efforts, i.e., controller joint torques) for the actuators based on operational space control to achieve the desired target (either pose or twist).
        """        
        # Get the Jacobian matrix for the end-effector.
        J = get_site_jac(self.model, self.data, self.eef_id)
        J = J[:, self.jnt_dof_ids]

        # Get the mass matrix and its inverse for the controlled degrees of freedom (DOF) of the robot.
        M_full = get_fullM(self.model, self.data)
        M = M_full[self.jnt_dof_ids, :][:, self.jnt_dof_ids]
        Mx, M_inv = task_space_inertia_matrix(M, J)

        # Get the joint velocities for the controlled DOF.
        dq = self.data.qvel[self.jnt_dof_ids]

        # Get the end-effector position, orientation matrix, and twist (spatial velocity).
        ee_pos = self.data.site_xpos[self.eef_id]
        ee_quat = r2q(self.data.site_xmat[self.eef_id].reshape(3, 3), order="xyzs")
        ee_pose = np.concatenate([ee_pos, ee_quat])
        ee_twist = J @ dq

        self.target_pose = target
        self.actual_pose = ee_pose
        
        # This is for the plots
        self.plot_data = np.concatenate((ee_pose, np.array(self.data.sensordata)))

        # Initialize the task space control signal (desired end-effector motion).
        u_task = np.zeros(6)

        if self.target_type == TargetType.POSE:
            # If the target type is pose, the target contains both position and orientation.
            target_pose = target
            target_twist = np.zeros(6)

            # Scale the task space control signal while ensuring it doesn't exceed the specified velocity limits.
            u_task = self._scale_signal_vel_limited(pose_error(ee_pose, target_pose))

        elif self.target_type == TargetType.TWIST:
            # If the target type is twist, the target contains the desired spatial velocity.
            target_twist = target

        else:
            raise ValueError("Invalid target type: {}".format(self.target_type))

        # Initialize the joint effort control signal (controller joint torques).
        u = np.zeros(self.n_joints)

        if np.all(target_twist == 0):
            # If the target twist is zero (no desired motion), apply damping to the controlled DOF.
            u -= self.kv * np.dot(M, dq)
        else:
            # If the target twist is not zero, calculate the task space control signal error.
            u_task += self.kv * (ee_twist - target_twist)

        # Compute the joint effort control signal based on the task space control signal.
        u -= np.dot(J.T, np.dot(Mx, u_task))

        # Compute the null space control signal to minimize the joint efforts in the null space of the task.
        u_null = np.dot(M, -self.null_damp_kv * dq)
        Jbar = np.dot(M_inv, np.dot(J.T, Mx))
        null_filter = np.eye(self.n_joints) - np.dot(J.T, Jbar.T)
        u += np.dot(null_filter, u_null)

        # Call the parent class's run method to apply the computed joint efforts to the robot actuators.
        super().run(u, ctrl)

    def _scale_signal_vel_limited(self, u_task: np.ndarray) -> np.ndarray:
        """
        Scale the control signal such that the arm isn't driven to move faster in position or orientation than the specified vmax values.

        Parameters:
            u_task (numpy.ndarray): The task space control signal.

        Returns:
            numpy.ndarray: The scaled task space control signal.
        """
        norm_xyz = np.linalg.norm(u_task[:3])
        norm_abg = np.linalg.norm(u_task[3:])
        scale = np.ones(6)
        if norm_xyz > self.sat_gain_xyz:
            scale[:3] *= self.scale_xyz / norm_xyz
        if norm_abg > self.sat_gain_abg:
            scale[3:] *= self.scale_abg / norm_abg

        return self.kv * scale * self.lamb * u_task
    
    # def target_reached(self):
    #     if self.actual_pose is not None and self.target_pose is not None:
    #         return max(np.abs(self.actual_pose - self.target_pose)) < self.target_tol
    #     else:
    #         return False
    def target_reached(self):
        if self.actual_pose is not None and self.target_pose is not None:
            return (max(np.abs(self.actual_pose[:3] - self.target_pose[:3])) < self.target_tol_pos and max(np.abs(self.actual_pose[-4:] - self.target_pose[-4:])) < self.target_tol_quat)
        else:
            return False

class AdmittanceController(OperationalSpaceController):
    def __init__(
        self,
        model: MjModel,
        data: MjData,
        model_names: MujocoModelNames,
        eef_name: str,
        joint_names: List[str],
        actuator_names: List[str],
        min_effort: List[float],
        max_effort: List[float],
        control_period: float,
        start_position = np.array([0, 0, 0]), 
        start_orientation = np.array([1.0, 0.0, 0.0, 0.0]),
        start_ft = np.array([0, 0, 0, 0, 0, 0]), 
        start_q = np.array([0, 0, 0, 0, 0, 0]),
        singularity_avoidance: bool = False,
        
    ) -> None:
        super().__init__(
            model, 
            data, 
            model_names, 
            eef_name, 
            joint_names, 
            actuator_names,
            min_effort, 
            max_effort,
            target_type=TargetType.POSE,
            kp=400.0,
            ko=400.0,
            kv=50.0,
            vmax_xyz=2,
            vmax_abg=2,
            null_damp_kv=10,
        )
        # super().__init__(
        #     model, 
        #     data, 
        #     model_names, 
        #     eef_name, 
        #     joint_names, 
        #     actuator_names, 
        #     min_effort, 
        #     max_effort, 
        # )
        self.control_period = control_period
        
        self.target_tol_pos = 0.008 #0.0075
        self.target_tol_quat = 0.015
        # TODO: INSERT MAGICAL CODE HERE

        # Gain matrices
        m = 1
        kenv = 20000 # 5000 for softbody
        kd = 3000 # 2500
        k = 10 # 100 # 4/m * kd - kenv

        self.M_tcp = np.array([[m,0,0],[0,m,0],[0,0,m]])
        self.K_tcp = np.array([[0,0,0],[0,k,0],[0,0,k]])
        self.D_tcp = np.array([[kd,0,0],[0,kd,0],[0,0,kd]])

        self._x_d = start_position
        
        #Initial conditions:
        self._x_d = np.array([0.0, 0.0, 0.0])
        self._dc_c = np.array([0.0, 0.0, 0.0])
        self._x_e = np.array([0.0, 0.0, 0.0])
        self._dx_e = np.array([0.0, 0.0, 0.0])
        self.target_force = np.array([0.0, 0.0, 0.0])

        self.point_cloud = Point_cloud()
        self.force_utils = Force_utils(self.model, self.data, self.model_names)
        self.transform_utils = Transform_utils(self.model, self.data, self.model_names)

        # Navigate to the parent directory of the script
        self.parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        ##########################
        # For orientational part #
        ##########################
        # self.Mo = np.diag([0.25, 0.25, 0.25])  # Orientation Mass
        # self.Ko = np.diag([0, 0, 0])  # Orientation Stiffness
        # self.Do = np.diag([5, 5, 5])  # Orientation Damping

        # self._quat_d = Rotation.from_quat(start_orientation)
        # self._omega_c = np.array([0.0, 0.0, 0.0])
        # self._quat_c = Rotation.from_quat(start_orientation)

        # self._omega_e = np.array([0.0, 0.0, 0.0])
        # self._quat_e = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])


    def admittance(self, target, i):
        tcp_rot_mat = self.data.site_xmat[self.model_names.site_name2id["tcp_site"]].reshape(3, 3)
        tcp_quat = r2q(tcp_rot_mat, order="xyzs")
        self.actual_pose = np.concatenate([self.data.site_xpos[self.model_names.site_name2id["tcp_site"]], tcp_quat])
        self.target_pose = target

        self._x_d = target[:3]

        if i == 0:
            self.target_force = np.matmul(tcp_rot_mat, np.array([0.0, 0.0, 0.0]))
        else:
            self.target_force = np.matmul(tcp_rot_mat, np.array([200.0, 0.0, 0.0])) # 10
        print("Target force: ", self.target_force)

        # Check for contact
        force, rot_contact, is_in_contact = self.force_utils._get_contact_info("belly") # obj options: "softbody" or "box"

        if is_in_contact:
            # self.target_force = np.array([-15.0, 0.0, 0.0])
            # self.target_force = np.matmul(tcp_rot_mat, np.array([-15.0, 0.0, 0.0]))

            tool_tip_pos = self.data.site_xpos[self.model_names.site_name2id["tcp_site"]]

            # surface_normal = -rot_contact[:, 0]
            surface_normal = self.point_cloud.get_surface_normal(tool_tip_point=tool_tip_pos, print_normal=False)
            self.align_rot_matrix = self.force_utils.align_with_surface_normal(surface_normal)

            # self.align_rot_matrix = self.force_utils._rotation_matrix_to_align_z_to_direction(-rot_contact[:, 0])
            # print(self.align_rot_matrix)

            target[-4:] = r2q(np.asarray(self.align_rot_matrix), order="xyzs")
        else:
            self.align_rot_matrix = q2r(target[-4:], order="xyzs")
        
        align_quaternion = target[-4:]
        
        # Update gains based on orientation function
        # self.M = self.align_rot_matrix @ self.M_tcp ########################
        # self.K = self.align_rot_matrix @ self.K_tcp ########################
        # self.D = self.align_rot_matrix @ self.D_tcp ########################

        # print(tcp_rot_mat)
        
        self.M = tcp_rot_mat @ self.M_tcp ########################
        self.K = tcp_rot_mat @ self.K_tcp ########################
        self.D = tcp_rot_mat @ self.D_tcp ########################

        # self.M = self.M_tcp
        # self.K = self.K_tcp
        # self.D = self.D_tcp

        # print(self.K)
        # print(self.K_tcp)

        # Positional part of the admittance controller
        # Step 1: Acceleration error
        print(- self.K @ self._x_e - self.D @ self._dx_e)
        ddx_e = np.linalg.inv(self.M) @ (-force + self.target_force - self.K @ self._x_e - self.D @ self._dx_e)

        # Step 2: Integrate -> velocity error
        self._dx_e += ddx_e * self.control_period # Euler integration

        # Step 3: Integrate -> position error
        self._x_e += self._dx_e * self.control_period # Euler integration

        # Step 4: Update the position
        self._x_c = self._x_d + self._x_e

        # print("Current Position: ", self.data.site_xpos[self.model_names.site_name2id["tcp_site"]])
        # print("Desired Position: ", self._x_d)
        print("Force: ", -force)
        # print("Position error: ", self._x_e)
        print("Compliant Position: ", self._x_c)

        print(self.actual_pose)
        print(self.target_pose)

        # with open(self.parent_dir + "/DATA_ROBOT.csv",'a') as fd:
        #     fd.write(f'{self.actual_pose[0]},{self.actual_pose[1]},{self.actual_pose[2]},{self.actual_pose[3]},{self.actual_pose[4]},{self.actual_pose[5]},{self.actual_pose[6]}\n')
        return self.transform_utils.tcp2eef(self._x_c, align_quaternion)



    def run(
        self, 
        target: np.ndarray,
        ctrl: np.ndarray,
        i = 0,
    ) -> None:
        # TODO: INSERT MAGICAL CODE HERE
        u = self.admittance(target, i)

        # Call the parent class's run method to apply the computed joint efforts to the robot actuators.
        super().run(u, ctrl)


    def target_reached(self):
        if self.actual_pose is not None and self.target_pose is not None:
            return (max(np.abs(self.actual_pose[:3] - self.target_pose[:3])) < self.target_tol_pos and max(np.abs(self.actual_pose[-4:] - self.target_pose[-4:])) < self.target_tol_quat)
        else:
            return False


    def admittance_orientational(self, target, wrench):
        #torque = self.data.sensordata[-3:]
        torque = wrench[-3:]

        self._quat_desired = Rotation.from_quat(target[-4:])

        # Rotational part of the compliance frame
        # Angular acceleration error
        domega_e = np.linalg.inv(self.Mo) @ (torque - self.Ko @ self._quat_e.as_quat()[1:] - self.Do @ self._omega_e)
        # Integrate -> angular velocity error
        # Compute q_next
        q_next = self.quatmultiply(self.quatExp(0.5 * self.control_period * domega_e), self._quat_e.as_quat())

        # Normalize q_next
        self._quat_e = Rotation.from_quat(q_next / np.linalg.norm(q_next))
        self._omega_e = self._omega_e + self.control_period * domega_e

        # multiply with the desired quaternion
        self._quat_c = self._quat_desired.as_quat() + self._quat_e.as_quat()
        
        u = [self._x_c[0], self._x_c[1], self._x_c[2], self._quat_c[0], self._quat_c[1], self._quat_c[2], self._quat_c[3]]
        return u
    

    def quaternion_to_euler(q):
        r = Rotation.from_quat(q)
        euler_angles = r.as_euler('xyz', degrees=True)
        return euler_angles
    

    def skew_symmetric(self, vector):
        x = vector[0]
        y = vector[1]
        z = vector[2]
        Sv = np.zeros((3, 3))
        Sv[1, 0] = z
        Sv[2, 0] = -y
        Sv[0, 1] = -z
        Sv[2, 1] = x
        Sv[0, 2] = y
        Sv[1, 2] = -x
        return Sv
    

    def quatExp(self, v):
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            return np.array([np.cos(v_norm), 0, 0, 0])
        else:
            return np.array([np.cos(v_norm), *np.sin(v_norm) * v / v_norm])


    def quatmultiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z])