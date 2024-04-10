from simulation.simulator import MJ
# from simulation.data_query import Data_Query 
# from robot.robot_control import Robot_Controler
# from time import sleep
# import roboticstoolbox as rtb
# import numpy as np
# from spatialmath import SE3
# import transformations as tf

def main() -> None:
  sim = MJ()
  sim.run()

if __name__ == "__main__":
  main()
  exit()


  controller = Robot_Controler()
  simulator = MJ(controller)
  controller.set_sim(simulator=simulator)
    # Universal Robot UR5e kiematics parameters 

  simulator.start()
  robot_data = Data_Query(simulator.d)
  fk = controller.forKin([-0.3, 0, -2.2, 0, 2, 0.7854])
  ik = controller.invKin(fk)
  simulator.sendJoint(ik.q)
  prev_tcp = controller.forKin(simulator.getState())
  while 1:
    input("press to continue")
  #   currentTCP = controller.forKin(simulator.getState())
  #   change = prev_tcp - currentTCP
  #   #print(change)
  #   prev_tcp = currentTCP
  #   target = tf.identity_matrix()
  #   target[:3, :3] = robot_data.directionToNormal(currentTCP.R, [0, 1, 0]) #currentTCP.R
  #   transform = currentTCP.t
  #   target[0, 3] = transform.T[0] 
  #   target[1, 3] = transform.T[1]
  #   target[2, 3] = transform.T[2] 

  #   config = controller.invKin(target)
  #   simulator.sendJoint(config.q)

    
  
    