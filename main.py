from simulation.simulator import MJ
import numpy as np
from colect_sim.env.ur5_env import UR5Env
from colect_sim.utils.traj_generation import linear_traj_w_gauss_noise

from robot.robot_control import Robot 

def main() -> None:
  env = UR5Env()

  quat = np.array([0,1,0,1])
  quat = quat / np.linalg.norm(quat)
  # Linear scanning near one edge
  traj_start = np.array([0.375, 0.45, 0.205,quat[0],quat[1],quat[2],quat[3]])
  traj_stop = np.array([0.625, 0.45, 0.205,quat[0],quat[1],quat[2],quat[3]])
  traj = linear_traj_w_gauss_noise(traj_start, traj_stop, 100, 0., 0.0005)
  
  i = 0
  terminated = False
  while not terminated:
      next = traj[i]
      op_target_reached = False
      while not op_target_reached:
          op_target_reached, terminated = env.step(next)
      env.enable_recording = True # inelegant, but works for aligning the recording to the target
      i += 1
      if i > len(traj) - 1 : terminated = True

if __name__ == "__main__":
  main()