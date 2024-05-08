import numpy as np

from enum import Enum
from colect_sim.controller.joint_effort_controller import JointEffortController
from colect_sim.utils.mujoco_utils import get_site_jac, get_fullM
from colect_sim.utils.transform_utils import mat2quat
from colect_sim.utils.controller_utils import task_space_inertia_matrix, pose_error
from colect_sim.utils.mujoco_utils import MujocoModelNames
from mujoco import MjModel, MjData
from typing import List

from add_utils.point_cloud import Point_cloud
from add_utils.force_utils import Force_utils
from scipy.spatial.transform import Rotation


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

        self.point_cloud = Point_cloud()
        self.force_utils = Force_utils(self.model, self.data, model_names)

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
        ee_quat = mat2quat(self.data.site_xmat[self.eef_id].reshape(3, 3))
        ee_pose = np.concatenate([ee_pos, ee_quat])
        ee_twist = J @ dq

        self.target_pose = target
        self.actual_pose = ee_pose
        
        # This is for the plots
        self.plot_data = np.concatenate((ee_pose, np.array(self.data.sensordata)))

        # Initialize the task space control signal (desired end-effector motion).
        u_task = np.zeros(6)




        # This is the tool tip position: self.data.site_xpos[self.model_names.site_name2id["tcp_site"]]
        # print(self.data.site_xpos[self.model_names.site_name2id["tcp_site"]])

        ###########################################
        # Option 1 to get force --> Contact force #
        ###########################################
        # force = self.force_utils._get_contact_info("box")
        # print("The contact force is: ", force)


        ##########################################
        # Option 2 to get force --> Sensor force #
        ##########################################
        # force = self.force_utils._get_sensor_force()
        # print("The sensor force is: ", force)

        # Save force data for further analysis
        # with open('wrench_data.txt', 'a') as file:
        #     np.savetxt(file, [force], fmt='%f')  # fmt


        #############################################################
        # Only get force if in contact and get alignment rot matrix #
        #############################################################
        is_in_contact, _ = self.force_utils._is_in_contact("softbody") # obj options: "softbody" or "box"

        if is_in_contact:
            force, eef_rot_mat = self.force_utils._get_sensor_force()
            force = -1.0 * force
            print(force)

            tool_tip_pos = self.data.site_xpos[self.model_names.site_name2id["tcp_site"]]
            surface_normal = self.point_cloud.get_surface_normal(tool_tip_point=tool_tip_pos, print_normal=False)

            # print(self.data.site_xmat[self.model_names.site_name2id["eef_site"]].reshape(3,3))

            new_eef_rot_mat = self.force_utils.align_with_surface_normal(eef_rot_mat, surface_normal)
            # print(new_eef_rot_mat)



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
    
    def target_reached(self):
        if self.actual_pose is not None and self.target_pose is not None:
            return max(np.abs(self.actual_pose - self.target_pose)) < self.target_tol
        else:
            return False


class AdmittanceController(JointEffortController):
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
        # super().__init__(
        #     model, 
        #     data, 
        #     model_names, 
        #     eef_name, 
        #     joint_names, 
        #     actuator_names,
        #     min_effort, 
        #     max_effort,
        #     target_type=TargetType.POSE,
        #     kp=400.0,
        #     ko=200.0,
        #     kv=50.0,
        #     vmax_xyz=2,
        #     vmax_abg=2,
        #     null_damp_kv=10,
        # )
        super().__init__(
            model, 
            data, 
            model_names, 
            eef_name, 
            joint_names, 
            actuator_names, 
            min_effort, 
            max_effort, 
        )
        self.control_period = control_period
        
        self.target_tol = 0.0075 #0.0075
        # TODO: INSERT MAGICAL CODE HERE

        # Gain matrices
        m = 22.5 # 1
        kenv = 20000 # 5000 for softbody
        kd = 85 # 1
        k = 0 # 4/m * kd - kenv

        self.M = np.array([[m,0,0],[0,m,0],[0,0,m]])
        self.K = np.array([[k,0,0],[0,k,0],[0,0,0]])
        self.D = np.array([[kd,0,0],[0,kd,0],[0,0,kd]])

        self._x_d = start_position
        
        #Initial conditions:
        self._dc_c = np.array([0.0, 0.0, 0.0])
        self._x_e = np.array([0.0, 0.0, 0.0])
        self._dx_e = np.array([0.0, 0.0, 0.0])
        self.target_force = np.array([0.0, 0.0, -150.0])

        self.point_cloud = Point_cloud()
        self.force_utils = Force_utils(self.model, self.data, self.model_names)

        self.contact = False

        #################################################################################################
        self.Mo = np.diag([0.25, 0.25, 0.25])  # Orientation Mass
        self.Ko = np.diag([0, 0, 0])  # Orientation Stiffness
        self.Do = np.diag([5, 5, 5])  # Orientation Damping

        self._quat_d = Rotation.from_quat(start_orientation)
        self._omega_c = np.array([0.0, 0.0, 0.0])
        self._quat_c = Rotation.from_quat(start_orientation)

        self._omega_e = np.array([0.0, 0.0, 0.0])
        self._quat_e = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])


    def admittance(self, target):
        # Check for contact
        is_in_contact, _ = self.force_utils._is_in_contact("box") # obj options: "softbody" or "box"

        if is_in_contact:
            self.contact = True

        if self.contact:
            self.Xc = self.data.site_xpos[self.model_names.site_name2id["eef_site"]] # MAYBE end-effector is not the correct position we want here
            print("Current Position: ", self.Xc)
            Xd = target[:3]
            print("Target Position: ", Xd)

            # force, eef_rot_mat = self.force_utils._get_sensor_force()
            force = self.force_utils._get_contact_info("box")
            print("Force", force)

            # tool_tip_pos = self.data.site_xpos[self.model_names.site_name2id["tcp_site"]]
            # surface_normal = self.point_cloud.get_surface_normal(tool_tip_point=tool_tip_pos, print_normal=False)
            # rot_align = self.force_utils.align_with_surface_normal(eef_rot_mat, surface_normal)
            
            # Update gains based on orientation function
            # M = self.M_prev #rot_align @ self.M_prev 
            # K = self.K_prev #rot_align @ self.K_prev
            # D = self.D_prev #rot_align @ self.D_prev

            self.Xe = self.Xc - Xd

            # Step 1: Calculate acceleration
            self.acc = np.linalg.inv(M) @ (force + self.target_force - D @ self.vel - K @ self.Xe)

            # Step 2: Integrate acceleration to get velocity
            self.vel = self.int_acc(self.acc, self.vel, self.dt)
            
            # Step 3: Integrate velocity to get position
            self.Xe = self.int_vel(self.vel, self.Xe, self.dt)
            # print("Position update: ", self.Xe)
            
            # Step 4: Update compliant position
            self.Xc = self.Xe + Xd
            print("Compliant Position: ", self.Xc)

            # align_quaternion = Rotation.from_matrix(rot_align).as_quat()

            return np.concatenate([self.Xc, target]) #align_quaternion[0], align_quaternion[1], align_quaternion[2], align_quaternion[3]])
        else:
            return target

    def int_acc(self, acc, vel, dt): # Euler integration
        vel = vel + acc * dt
        return vel

    def int_vel(self, vel, pos, dt): # Euler integration
        pos = pos + vel * dt
        return pos
    

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
    

    def run(
        self, 
        target: np.ndarray,
        ctrl: np.ndarray,
    ) -> None:
        
        # TODO: INSERT MAGICAL CODE HERE
        self._x_c = self.data.site_xpos[self.eef_id]
        eef_quat = mat2quat(self.data.site_xmat[self.eef_id].reshape(3, 3))
        self.actual_pose = np.concatenate([self._x_c, eef_quat])
        self.target_pose = target

        self._x_d = target[:3]
        self._quat_desired = Rotation.from_quat(target[-4:])

        # Get current contact/sensor force
        # force, eef_rot_mat = self.force_utils._get_sensor_force()
        force = self.force_utils._get_contact_info("box")
        torque = self.data.sensordata[-3:]

        # Positional part of the compliance frame
        # Acceleration error
        ddx_e = np.linalg.inv(self.M) @ (force - self.K @ self._x_e - self.D @ self._dx_e)
        # Integrate -> velocity error
        self._dx_e += ddx_e * self.control_period
        # Integrate -> position error
        self._x_e += self._dx_e * self.control_period
        # Update the position
        x_c = self._x_d + self._x_e



        # # Rotational part of the compliance frame
        # # Angular acceleration error
        # domega_e = np.linalg.inv(self.Mo) @ (torque - self.Ko @ self._quat_e.as_quat()[1:] - self.Do @ self._omega_e)
        # # Integrate -> angular velocity error
        # # Compute q_next
        # q_next = self.quatmultiply(self.quatExp(0.5 * self.control_period * domega_e), self._quat_e.as_quat())

        # # Normalize q_next
        # self._quat_e = Rotation.from_quat(q_next / np.linalg.norm(q_next))
        # self._omega_e = self._omega_e + self.control_period * domega_e

        # # multiply with the desired quaternion
        # self._quat_c = self._quat_desired.as_quat() + self._quat_e.as_quat()
        # #quat_c_arr = quaternion.as_float_array(self._quat_c)
        # rotation_obj = Rotation.from_quat(self._quat_c)
        # # Convert the rotation object to a rotation vector (Euler angle)
        # eul_ang = rotation_obj.as_rotvec()


        target_quat = target[3:]
        r = Rotation.from_quat(target_quat)
        target_eul_ang = r.as_euler('zyx', degrees=True)
        print(target_eul_ang)
        u_task = [x_c[0], x_c[1], x_c[2], target_eul_ang[0], target_eul_ang[1], target_eul_ang[2]]
        #u_task = [x_c[0], x_c[1], x_c[2], eul_ang[0], eul_ang[1], eul_ang[2]]

        J_full = get_site_jac(self.model, self.data, self.eef_id)
        J = J_full[:, self.jnt_dof_ids]
        # Compute the joint effort control signal based on the task space control signal
        #u_task = self.spatial_controller(u_task, self.control_period)
        u = np.dot(np.linalg.pinv(J), u_task)





        # self.Xc = self.data.site_xpos[self.model_names.site_name2id["eef_site"]] # MAYBE end-effector is not the correct position we want here
        # eef_quat = mat2quat(self.data.site_xmat[self.eef_id].reshape(3, 3))
        # self.eef_pose = np.concatenate([self.Xc, eef_quat])
        # self.target = target

        # u = self.admittance(target)


        # Call the parent class's run method to apply the computed joint efforts to the robot actuators.
        super().run(u, ctrl)


    def target_reached(self):
        if self.actual_pose is not None and self.target_pose is not None:
            return max(np.abs(self.actual_pose - self.target_pose)) < self.target_tol
        else:
            return False
