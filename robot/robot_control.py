# from simulation.simulator import MJ
# from simulation.data_query import Data_Query
import roboticstoolbox as rtb
from spatialmath import SE3
import numpy as np
from roboticstoolbox import IKSolution

from typing import List, Union


class Robot:
    def __init__(self, m, d):
        # Universal Robot UR5e kiematics parameters 
        self.robot = rtb.DHRobot(
            [ 
                rtb.RevoluteDH(d=0.1625, alpha = np.pi/2),
                rtb.RevoluteDH(a=-0.425),
                rtb.RevoluteDH(a=-0.3922),
                rtb.RevoluteDH(d=0.1333, alpha=np.pi/2),
                rtb.RevoluteDH(d=0.0997, alpha=-np.pi/2),
                rtb.RevoluteDH(d=0.0996)
            ], name="UR5e"
        )

        self._m = m
        self._d = d

        self._HOME = [np.pi, -np.pi/2.0, np.pi/2.0, -np.pi/2.0, -np.pi/2.0, 0.0]

    def get_robot(self):
        return self.robot

    def sendJoint(self, join_values):
        with self.simulator.jointLock:
          for i in range(0, 6):
            self.simulator.joints[i] = join_values[i]
          self.sendPositions = True

    def forKin(self, q):
        return self.robot.fkine(q)

    @property
    def n_actators(self) -> int:
        return 6

    def invKin(self, desiredTCP):
          # IKSolution contians q, success, iteration, searches, residual
          ik_sol: IKSolution = self.robot.ikine_LM(desiredTCP, self.get_q)
          return ik_sol.q
    
    def get_q(self) -> list[float]:
        return [self._d.joint(f"joint{i+1}").qpos[0] for i in range(self.n_actators)]
    
    def set_q(self, q) -> None:
        for i in range(self.n_actators):
            self._d.actuator(f"actuator{i+1}").ctrl = q[i]

    def home(self) -> None:
        self.set_q(q=self._HOME)

    def set_ee_pose(self, T: SE3):
        q = self.invKin(T)
        self.set_q(q=q)

    def set_ee_pose_compared(self, 
            pose: Union[None, List[float], np.ndarray, SE3],
            ) -> None:

        if isinstance(pose, SE3):
            target_pose = pose
        else:
            # Assuming pose is a list or numpy array [x, y, z, qw, qx, qy, qz]
            target_pose = SE3(pose[:3], pose[3:])

        q_sols = []
        for _ in range(10):
            ik_sol: IKSolution = self.robot.ikine_LM(Tep=target_pose)
            q_sol, success, iterations, searches, residual = ik_sol.q, ik_sol.success, ik_sol.iterations, ik_sol.searches, ik_sol.residual
            q_sols.append( (q_sol, success, iterations, searches, residual) )
        
        if not np.any([x[1] for x in q_sols]):
            raise ValueError(f"Inverse kinematics failed to find a solution to ee pose. Tried {10} times. [INFO]: \n\t{q_sol=}\n\t{success=}\n\t{iterations=}\n\t{searches=}\n\t{residual=}")

        q0 = self.get_q()

        lengths = []
        for i in range(len(q_sols)):
            q_end = q_sols[i][0]
            # traj = rtb.jtraj(q0 = q0, qf = q_end, t = 50)
            lenght = np.linalg.norm(q_end - q0)
            lengths.append((lenght,i))

        best_i = min(lengths)[1]

        self.set_q(q_sols[best_i][0])

    def get_ee_pose(self) -> SE3:
        return self.forKin(self.get_q())