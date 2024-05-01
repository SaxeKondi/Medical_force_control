import numpy as np
from colect_sim.env.ur5_env import UR5Env
from colect_sim.utils.traj_generation import linear_traj_w_gauss_noise


def main() -> None:
  env = UR5Env()

  quat = np.array([0,1,0,1])
  quat = quat / np.linalg.norm(quat)
  # Linear scanning near one edge
  traj_start = np.array([0.375, 0.46, 0.205,quat[0],quat[1],quat[2],quat[3]])
  traj_stop = np.array([0.625, 0.45, 0.205,quat[0],quat[1],quat[2],quat[3]])

  # Move in neg z direction
  # traj_start = np.array([0.5, 0.45, 0.225,quat[0],quat[1],quat[2],quat[3]])
  # traj_stop = np.array([0.5, 0.45, -5,quat[0],quat[1],quat[2],quat[3]])

  # Move in pos x direction
  # traj_start = np.array([0.35, 0.45, 0.21,quat[0],quat[1],quat[2],quat[3]])
  # traj_stop = np.array([0.5, 0.45, 0.21,quat[0],quat[1],quat[2],quat[3]])

  # Move in neg x direction
  # traj_start = np.array([0.68, 0.45, 0.21,quat[0],quat[1],quat[2],quat[3]])
  # traj_stop = np.array([0.5, 0.45, 0.21,quat[0],quat[1],quat[2],quat[3]])

  # Move in pos y direction
  # traj_start = np.array([0.5, 0.35, 0.21,quat[0],quat[1],quat[2],quat[3]])
  # traj_stop = np.array([0.5, 0.45, 0.21,quat[0],quat[1],quat[2],quat[3]])

  traj = linear_traj_w_gauss_noise(traj_start, traj_stop, 100, 0., 0.0005)

  i = 0
  terminated = False
  while not terminated:
      next = traj[i]
      op_target_reached = False
      while not op_target_reached:
          op_target_reached, terminated = env.step(next, controller_name="admittance") # Controller options: "op_space" or "admittance"
      env.enable_recording = True # inelegant, but works for aligning the recording to the target
      i += 1
      if i > len(traj) - 1 : terminated = True

if __name__ == "__main__":
  main()