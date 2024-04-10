import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def linear_traj_w_gauss_noise(start: np.ndarray, end: np.ndarray, n_points: int = 100, mean: float = 0.05, var: float = 0.05):
    """
    Generate a linear trajectory with Gaussian noise between two poses.

    Parameters:
    - start: Starting pose as a numpy array [position, quaternion].
    - end: Ending pose as a numpy array [position, quaternion].
    - n_points: Number of points in the trajectory, including start and end (default is 100).
    - mean: Mean of the Gaussian noise (default is 0.05).
    - var: Variance of the Gaussian noise (default is 0.05).

    Returns:
    - traj: List of poses along the trajectory, each pose represented as [position, quaternion].
    """

    # Extract position and quaternion from start and end poses
    start_pos, start_quat = start[:3], start[3:]
    end_pos, end_quat = end[:3], end[3:]

    # Generate linear trajectory between start and end
    positions = [start_pos + i/n_points * (end_pos - start_pos) for i in range(n_points)]
    start_rot = Rotation.from_quat(start_quat)
    end_rot = Rotation.from_quat(end_quat)
    key_rots = Rotation.concatenate([start_rot, end_rot])
    key_times = [0,1]
    slerp = Slerp(key_times, key_rots)
    rotations = slerp(np.arange(n_points)/n_points)

    # Add Gaussian noise to positions and orientations
    noise_positions = np.random.normal(loc=0, scale=var, size=(n_points, 3)) + mean
    noise_orientations = np.random.normal(loc=0, scale=var, size=(n_points, 4)) + mean

    # Apply noise to positions and orientations
    positions += noise_positions
    rotations = Rotation.as_quat(rotations)
    rotations += noise_orientations

    # Combine positions and orientations to form the trajectory
    traj = np.column_stack((positions, rotations))

    return traj

if __name__=='__main__':
    quat = np.array([0,1,0,1])
    quat = quat / np.linalg.norm(quat)
    start_pose = np.array([0.375, 0.45, 0.205,quat[0],quat[1],quat[2],quat[3]])
    end_pose = np.array([0.625, 0.45, 0.205,quat[0],quat[1],quat[2],quat[3]])
    trajectory = linear_traj_w_gauss_noise(start_pose, end_pose, n_points=100, mean=0.0001, var=0.0001)
    print(trajectory)