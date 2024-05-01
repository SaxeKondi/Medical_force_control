import numpy as np

from enum import Enum
from colect_sim.controller.joint_effort_controller import JointEffortController
from colect_sim.controller.joint_velocity_controller import JointVelocityController
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
        self.target_tol = 0.0075#0.01
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
            ko=200.0,
            kv=50.0,
            vmax_xyz=2,
            vmax_abg=2,
            null_damp_kv=10,
        )
        
        self.target_tol = 0.0075 #0.0075
        # TODO: INSERT MAGICAL CODE HERE

        # Gain matrices
        m1 = 3
        m2 = 3
        m3 = 3
        k1 = 10
        k2 = 10
        k3 = 10
        kenv1 = 100 #set by user depending on current object to be reconstructed
        kenv2 = 100
        kd1 = 2*np.sqrt(m1*(k1+kenv1))
        kd2 = 2*np.sqrt(m2*(k2+kenv2))

        self.M_prev = np.array([[m1,0,0],[0,m2,0],[0,0,m3]])
        self.K_prev = np.array([[k1,0,0],[0,k2,0],[0,0,0]])  #3 element of 3rd row can be zero
        self.D_prev = np.array([[kd1,0,0],[0,kd2,0],[0,0,k3]])

        # Other parameters
        self.dt = 0.001 #Based on the control loop of the robot (given by simulation). 1/500 in real UR5 environment. 
        
        #Initial conditions:
        self.vel = np.array([0, 0, 0])

        self.Xc = data.site_xpos[model_names.site_name2id["eef_site"]] # MAYBE end-effector is not the correct position we want here
        self.Xe = np.array([0, 0, 0])

        self.target_force = np.array([0, 0, 0])

        self.point_cloud = Point_cloud()
        self.force_utils = Force_utils(self.model, self.data, self.model_names)

        self.contact = False


    def admittance(self, target):
        # Check for contact
        is_in_contact, _ = self.force_utils._is_in_contact("box") # obj options: "softbody" or "box"

        if is_in_contact:
            self.contact = True

        if self.contact:
            self.Xc = self.data.site_xpos[self.model_names.site_name2id["eef_site"]] # MAYBE end-effector is not the correct position we want here
            Xd = target[:3]

            force, eef_rot_mat = self.force_utils._get_sensor_force()
            # force = -1.0 * force
            # print(force)

            tool_tip_pos = self.data.site_xpos[self.model_names.site_name2id["tcp_site"]]
            surface_normal = self.point_cloud.get_surface_normal(tool_tip_point=tool_tip_pos, print_normal=False)

            # print(self.data.site_xmat[self.model_names.site_name2id["eef_site"]].reshape(3,3))

            rot_align = self.force_utils.align_with_surface_normal(eef_rot_mat, surface_normal)
            
            M = rot_align @ self.M_prev #update gains based on orientation function
            K = rot_align @ self.K_prev
            D = rot_align @ self.D_prev
            
            self.Xe = self.Xc - Xd
            # print(self.Xc)
            # print("Xd", Xd)

            # print("Type of wrench:", force.shape)
            # print("Type of target force:", self.target_force.shape)
            # print("Type of vel:", self.vel.shape)
            # print("Type of pos errr:", self.Xe.shape)
            # print("Type of K:", K.shape)
            # print("Type of D:", D.shape)
            # print("Type of M:", M.shape)

            # Step 1: Calculate acceleration
            self.acc = np.linalg.inv(M) @ (force + self.target_force - D @ self.vel - K @ self.Xe)

            # Step 2: Integrate acceleration to get velocity
            self.vel = self.int_acc(self.acc, self.vel, self.dt)
            
            # Step 3: Integrate velocity to get position
            self.Xe = self.int_vel(self.vel, self.Xe, self.dt)

            # print(self.Xe)
            # print("Xd", Xd)
            
            # Step 4: Update current position
            self.Xc = self.Xe + Xd
            print(self.Xc)

            align_quaternion = Rotation.from_matrix(rot_align).as_quat()
            # print([self.Xc[0], self.Xc[1], self.Xc[2], align_quaternion[0], align_quaternion[1], align_quaternion[2], align_quaternion[3]])
            return np.array([self.Xc[0], self.Xc[1], self.Xc[2], align_quaternion[0], align_quaternion[1], align_quaternion[2], align_quaternion[3]]) #np.array([self.Xc, Rotation.from_matrix(rot_align).as_quat()]) #self.tool_to_base(self.Xc)
        else:
            return target


    def int_acc(self, acc, vel, dt):
        vel = vel + acc * dt
        return vel


    def int_vel(self, vel, pos, dt):
        pos = pos + vel * dt
        return pos


    def run(
        self, 
        target: np.ndarray,
        ctrl: np.ndarray,
    ) -> None:
        
        # TODO: INSERT MAGICAL CODE HERE
        u = self.admittance(target)

        # Call the parent class's run method to apply the computed joint efforts to the robot actuators.
        super().run(u, ctrl)  

    def target_reached(self):
        if self.actual_pose is not None and self.target_pose is not None:
            return max(np.abs(self.actual_pose - self.target_pose)) < self.target_tol
        else:
            return False


class ParallelForcePositionController(JointVelocityController):
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
        min_velocity: List[float],
        max_velocity: List[float],
        kp_jnt_vel: List[float], 
        ki_jnt_vel: List[float], 
        kd_jnt_vel: List[float], 
        kp: np.ndarray,
        ki: np.ndarray,
        kd: np.ndarray,
        control_period: float,
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
            min_velocity,
            max_velocity,
            kp_jnt_vel,
            ki_jnt_vel,
            kd_jnt_vel,
        )
        # TODO: INSERT MAGICAL CODE HERE

        self.target_tol = 0.0075

    def run(
        self, 
        target_pose: np.ndarray,
        ctrl: np.ndarray,
    ) -> None:

       # TODO: INSERT MAGICAL CODE HERE

        # Call the parent class's run method to apply the computed joint efforts to the robot actuators.
        super().run(u, ctrl)  

    def target_reached(self):
        if self.actual_pose is not None and self.target_pose is not None:
            return max(np.abs(self.actual_pose - self.target_pose)) < self.target_tol
        else:
            return False