import numpy as np
import os

# Get the directory path of the current Python script
script_dir = os.path.dirname(__file__)

# Navigate to the parent directory of the script
parent_dir = os.path.dirname(script_dir)

# Read data from file
data = np.loadtxt(parent_dir + "\data\softbody_pos_1.txt")

# Calculate the minimum and maximum values of each column
min_values = np.min(data, axis=0)
max_values = np.max(data, axis=0)

# Calculate the minimum and maximum values of each dimension (x, y, z)
min_x, min_y, min_z = np.min(data, axis=0)
max_x, max_y, max_z = np.max(data, axis=0)

# Calculate the center coordinates
center_x = (min_x + max_x) / 2
center_y = (min_y + max_y) / 2
center_z = (min_z + max_z) / 2

# Print the center coordinates
print(f"Center of the square: ({center_x}, {center_y}, {center_z})")

print(f"Length: {max_x - min_x}")
print(f"Width: {max_y - min_y}")
print(f"Height: {max_z - min_z}")