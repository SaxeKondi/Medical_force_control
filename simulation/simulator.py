import time
from threading import Thread, Lock
import mujoco
import mujoco.viewer
import numpy as np
import roboticstoolbox as rtb
import time
import glfw
from robot.robot_control import Robot
from spatialmath import SE3
from utils import utility 

class MJ:
  def __init__(self):
    self.m = mujoco.MjModel.from_xml_path('scene/scene.xml')
    self.d = mujoco.MjData(self.m)
    print(self.d.get_body_xpos('softbody'))
    self._data_lock = Lock()
    self.robot = Robot(m=self.m,d=self.d)
    
  def run(self) -> None:
    self.th = Thread(target=self.launch_mujoco, daemon=True)
    self.th.daemon = True
    self.th.start()
    input()
    print("done...")
 
  def key_cb(self, key):
    """
    Function for debugging. 
    space should make the robot stay where it is
    , takes the robot to home position
    . prints the end effector pose
    """
    if key == glfw.KEY_SPACE:
      T_curr = self.robot.get_ee_pose()
      T_des = T_curr @ SE3.Ty(0)
      self.robot.set_ee_pose_compared(T_des)

    if key == glfw.KEY_COMMA:
      self.robot.home()

    if key == glfw.KEY_PERIOD:
      print("ee pose = \n", self.robot.get_ee_pose())

    if key ==  glfw.KEY_F:
      print("Force: ", self.d.sensordata)

    if key == glfw.KEY_A:
      # Align to force
      pose = self.robot.get_ee_pose()
      print("current pose: ", pose)
      r = utility.directionToNormal(
        pose.R,
        self.d.sensordata[:3]  
      )
      rotated_pose = SE3.Rt(r, pose.t)
      print("changed pose: ", rotated_pose)
      self.robot.set_ee_pose_compared(rotated_pose)

  def launch_mujoco(self):
    with mujoco.viewer.launch_passive(self.m, self.d, key_callback=self.key_cb) as viewer:
      while viewer.is_running():
        step_start = time.time()
        with self._data_lock:
          mujoco.mj_step(self.m, self.d)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
          time.sleep(time_until_next_step)
  
