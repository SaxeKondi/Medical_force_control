import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import time

from colect_sim.controller.operational_space_controller import ParallelForcePositionController, AdmittanceController, OperationalSpaceController, TargetType
from colect_sim.env.mujoco_env import MujocoEnv
from colect_sim.utils.mujoco_utils import MujocoModelNames
from mujoco import viewer
from os import path

from threading import Thread, Lock

np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})


DEFAULT_CAMERA_CONFIG = {
    "distance": 2.2,
    "azimuth": 0.0,
    "elevation": -20.0,
    "lookat": np.array([0, 0, 1]),
}

class UR5Env(MujocoEnv):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 1000, # Basically no fps limit
    }

    def __init__(
        self,
        model_path="../../scene/scene.xml",
        frame_skip=1,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        plot=False,
        **kwargs,
    ):
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            model_path,
        )

        super().__init__(
            xml_file_path,
            frame_skip,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.init_qvel = self.data.qvel.copy()
        self.init_ctrl = self.data.ctrl.copy()

        self.model_names = MujocoModelNames(self.model) 

        self.invdyn_controller = OperationalSpaceController(
            model=self.model, 
            data=self.data, 
            model_names=self.model_names,
            eef_name='eef_site', 
            joint_names=[
                'shoulder_pan_joint',
                'shoulder_lift_joint',
                'elbow_joint',
                'wrist_1_joint',
                'wrist_2_joint',
                'wrist_3_joint',
            ],
            actuator_names=[
                'shoulder_pan',
                'shoulder_lift',
                'elbow',
                'wrist_1',
                'wrist_2',
                'wrist_3',
            ],
            min_effort=[-150, -150, -150, -150, -150, -150],
            max_effort=[150, 150, 150, 150, 150, 150],
            target_type=TargetType.POSE,
            kp=400.0,
            ko=200.0,
            kv=50.0,
            vmax_xyz=2,
            vmax_abg=2,
            null_damp_kv=10,
        )
        
        self.adm_controller = AdmittanceController(
            model=self.model, 
            data=self.data, 
            model_names=self.model_names,
            eef_name='eef_site', 
            joint_names=[
                'shoulder_pan_joint',
                'shoulder_lift_joint',
                'elbow_joint',
                'wrist_1_joint',
                'wrist_2_joint',
                'wrist_3_joint',
            ],
            actuator_names=[
                'shoulder_pan',
                'shoulder_lift',
                'elbow',
                'wrist_1',
                'wrist_2',
                'wrist_3',
            ],
            min_effort=[-150, -150, -150, -150, -150, -150],
            max_effort=[150, 150, 150, 150, 150, 150],
            control_period=self.model.opt.timestep,
        )

        self.pf_controller = ParallelForcePositionController(
            model=self.model, 
            data=self.data, 
            model_names=self.model_names,
            eef_name='eef_site', 
            joint_names=[
                'shoulder_pan_joint',
                'shoulder_lift_joint',
                'elbow_joint',
                'wrist_1_joint',
                'wrist_2_joint',
                'wrist_3_joint',
            ],
            actuator_names=[
                'shoulder_pan',
                'shoulder_lift',
                'elbow',
                'wrist_1',
                'wrist_2',
                'wrist_3',
            ],
            min_effort=[-150, -150, -150, -150, -150, -150],
            max_effort=[150, 150, 150, 150, 150, 150],
            min_velocity=[-1, -1, -1, -1, -1, -1],
            max_velocity=[1, 1, 1, 1, 1, 1],
            kp_jnt_vel=[100, 100, 100, 100, 100, 100],
            ki_jnt_vel=0,
            kd_jnt_vel=0,
            kp=[10, 10, 10, 10, 10, 10],
            ki=[0, 0, 0, 0, 0, 0],
            kd=[3, 3, 3, 8, 8, 8],
            control_period=self.model.opt.timestep,
        )
        self.pf_controller.Kp_f = np.eye(3)
        self.pf_controller.Kv_f = np.eye(3)
        self.pf_controller.Kp_p = np.eye(3)
        self.pf_controller.Kd_p = np.eye(3)

        self.controller = self.invdyn_controller

        self.init_qpos_config = {
            "shoulder_pan_joint": np.pi / 2.0,
            "shoulder_lift_joint": -np.pi / 2.0,
            "elbow_joint": -np.pi / 2.0,
            "wrist_1_joint": -np.pi / 2.0,
            "wrist_2_joint": np.pi / 2.0,
            "wrist_3_joint": np.pi / 2.0,
        }
        for joint_name, joint_pos in self.init_qpos_config.items():
            joint_id = self.model_names.joint_name2id[joint_name]
            qpos_id = self.model.jnt_qposadr[joint_id]
            self.init_qpos[qpos_id] = joint_pos

        self.op_target = None
        
        home_data = [-0.33357, 0.81420, 0.06377, 0.00005, 0.70715, -0.00005, 0.70707, 0, 0, 0, 0, 0, 0]

        # For live plotting
        self.history_lock = Lock()
        self.history = [home_data]*2000

        self.plot = plot
        if self.plot:
            # Start the thread for live plotting
            self.p = Thread(target=self.runGraph, daemon=True)
            self.p.start()

        # This will hold recorded poses/forces
        self.recorded_data = []
        self.enable_recording = False

        # Go to the home pose
        self.reset_model()

        # Start the viewer
        self.viewer = viewer.launch_passive(self.model, self.data)
        self.wait_for_viewer()

    def runGraph(self):
        # Create figure for plotting
        fig = plt.figure(figsize=(10,6))
        gs = gridspec.GridSpec(2,14,wspace=3)

        
        dt = 0.001
        time = dt*np.arange(0, 2000)
        data = []
        lines = []
        labels = ['Position x [m]',
                  'Position y [m]',
                  'Position z [m]',
                  'Quaternion x [m]',
                  'Quaternion y [m]',
                  'Quaternion z [m]',
                  'Quaternion w [m]',
                  'Force x [N]',
                  'Force y [N]',
                  'Force z [N]',
                  'Torque x [Nmm]',
                  'Torque y [Nmm]',
                  'Torque z [Nmm]']
        
        axes = []
        for i in range(0, 13):
            if i < 7:
                ax = plt.subplot(gs[0, 2 * i:2 * i + 2])
            else:
                ax = plt.subplot(gs[1, 2 * i - 13:2 * i + 2 - 13])
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(labels[i])
            ax.grid()
            axes.append(ax)

        for i in range(7):
            data = [pose[i] for pose in self.history]
            line, = axes[i].plot(time, data)
            lines.append(line)
        for i in range(6):
            data = [wrench[7 + i] for wrench in self.history]
            line, = axes[7+i].plot(time, data)
            lines.append(line)
            
        def animate(i, plot_data):
            # Update line with new Y values
            with self.history_lock:
                for i in range(7):
                    data = [pose[i] for pose in plot_data]
                    lines[i].set_ydata(data[-2000:])
                    buffer = 0.1*(max(data) - min(data))
                    axes[i].set_ylim(min(data) - buffer, max(data) + buffer)
                for i in range(6):
                    data = [wrench[7 + i] for wrench in plot_data]
                    lines[7 + i].set_ydata(data[-2000:])
                    buffer = 0.1*(max(data) - min(data))
                    axes[7+i].set_ylim(min(data) - buffer, max(data) + buffer)

            return [line for line in lines]
        

        # Set up plot to call animate() function periodically

        ani = animation.FuncAnimation(fig,
            animate,
            fargs=(self.history, ),
            interval=50,
            blit=True,
            cache_frame_data=False)
        plt.show()

    def wait_for_viewer(self):
        timeout = 5.0
        start_time = time.time()
        while not self.viewer.is_running():
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print("Timeout while waiting for viewer to start.")

    def step(self, action):
        
        # Step the simulation
        ctrl = self.data.ctrl.copy()
        self.controller.run(
            action, 
            ctrl
        )
        self.do_simulation(ctrl, n_frames=1)

        # Update the visualization
        self.viewer.sync()

        # Update the live plotting
        if self.plot:
            with self.history_lock:
                if self.controller.plot_data is not None:
                    self.history.append(self.controller.plot_data)
                    self.history.pop(0)

        # Update the pose/force recording
        if self.enable_recording:
            self.recorded_data.append(self.controller.plot_data)

        return self.controller.target_reached(), not self.viewer.is_running()

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

    def close(self):
        if self.viewer.is_running():
            self.viewer.close()
        if self.plot:
            self.p.join(timeout=5.0)
