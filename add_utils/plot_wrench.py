import numpy as np
import matplotlib.pyplot as plt
import os

# Get the directory path of the current Python script
script_dir = os.path.dirname(__file__)

# Navigate to the parent directory of the script
parent_dir = os.path.dirname(script_dir)

# Read data from file
data = np.genfromtxt(parent_dir + '\data\wrench_data_fullrun.txt')

# Transpose the data to have columns as variables
data_transposed = data.T

# Calculate time array
time = np.arange(len(data)) / 500.0  # Sampling frequency is 500 Hz

# Plot each component
num_components = data_transposed.shape[0]

for i in range(num_components):
    plt.figure(figsize=(8, 4))
    plt.plot(time, data_transposed[i])
    plt.xlabel('Time (s)')
    plt.ylabel('Component {}'.format(i+1))
    plt.title('Component {} vs. Time'.format(i+1))
    plt.grid(True)
plt.show()
