import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Get the directory path of the current Python script
script_dir = os.path.dirname(__file__)

# Navigate to the parent directory of the script
parent_dir = os.path.dirname(script_dir)

# Adjust the path to your actual CSV file location
# file_path = parent_dir + '/DATA_ROBOT_hacked.csv'
file_path = parent_dir + '/DATA_ROBOT_real.csv'

# Read the CSV file
data = pd.read_csv(file_path, header=None, names=['x', 'y', 'z', 'q0', 'q1', 'q2', 'q3', 'f0', 'f1', 'f2', 'tf0', 'tf1', 'tf2'])

# Create a time vector based on the number of samples and the time step
time_step = 0.0005  # time step in seconds
time_vector = np.arange(len(data)) * time_step

# Plotting the coordinates and quaternion components
plt.figure(figsize=(14, 7))

# Plotting the 'x' coordinate
plt.subplot(1, 3, 1)
plt.plot(time_vector, data['x'])
plt.title('TCP Position - X Coordinate')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')

# Plotting the 'y' coordinate
plt.subplot(1, 3, 2)
plt.plot(time_vector, data['y'])
plt.ylim(0, 5)
plt.title('TCP Position - Y Coordinate')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')

# Plotting the 'z' coordinate
plt.subplot(1, 3, 3)
plt.plot(time_vector, data['z'])
plt.title('TCP Position - Z Coordinate')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')


# # Plotting the 'x' coordinate
# plt.subplot(2, 4, 1)
# plt.plot(time_vector, data['x'], label='X Coordinate')
# plt.title('TCP Position - X Coordinate')
# plt.xlabel('Time (s)')
# plt.legend()

# # Plotting the 'y' coordinate
# plt.subplot(2, 4, 2)
# plt.plot(time_vector, data['y'], label='Y Coordinate')
# plt.ylim(0, 5)
# plt.title('TCP Position - Y Coordinate')
# plt.xlabel('Time (s)')
# plt.legend()

# # Plotting the 'z' coordinate
# plt.subplot(2, 4, 3)
# plt.plot(time_vector, data['z'], label='Z Coordinate')
# plt.title('TCP Position - Z Coordinate')
# plt.xlabel('Time (s)')
# plt.legend()

# # Plotting the quaternion components
# plt.subplot(2, 4, 5)
# plt.plot(time_vector, data['q0'], label='Quaternion q0')
# plt.title('Quaternion Component q0')
# plt.xlabel('Time (s)')
# plt.legend()

# plt.subplot(2, 4, 6)
# plt.plot(time_vector, data['q1'], label='Quaternion q1')
# plt.title('Quaternion Component q1')
# plt.xlabel('Time (s)')
# plt.legend()

# plt.subplot(2, 4, 7)
# plt.plot(time_vector, data['q2'], label='Quaternion q2')
# plt.title('Quaternion Component q2')
# plt.xlabel('Time (s)')
# plt.legend()

# plt.subplot(2, 4, 8)
# plt.plot(time_vector, data['q3'], label='Quaternion q3')
# plt.title('Quaternion Component q3')
# plt.xlabel('Time (s)')
# plt.legend()

# Show plot for coordinates and quaternions
plt.tight_layout()
plt.show()

# Plotting forces and target forces in a separate figure
plt.figure(figsize=(14, 7))

# Plotting forces and target forces
plt.subplot(3, 1, 1)
plt.plot(time_vector, data['f0'], label='Force')
plt.plot(time_vector, data['tf0'], label='Target Force', linestyle='--')
plt.title('Force and Target Force in X Direction')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time_vector, data['f1'], label='Force')
plt.plot(time_vector, data['tf1'], label='Target Force', linestyle='--')
plt.title('Force and Target Force in Y Direction')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time_vector, data['f2'], label='Force')
plt.plot(time_vector, data['tf2'], label='Target Force', linestyle='--')
plt.title('Force and Target Force in Z Direction')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()

# Show plot for forces and target forces
plt.tight_layout()
plt.show()
