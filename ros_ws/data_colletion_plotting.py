import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the merged CSV for controller 1
df1 = pd.read_csv('/workspace/ros_ws/panda_ee_full_20250505_175652.csv')

## Load csv for baseline

##load csv for 

# Extract x, y, z position columns from controller 1
x_real_1 = df1['ee_pose_field.position.x']
y_real_1 = df1['ee_pose_field.position.y']
z_real_1 = df1['ee_pose_field.position.z']


# Create 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_real_1, y_real_1, z_real_1, label='End-Effector Trajectory', color='blue')

# Axis labels and title
ax.set_title("3D End-Effector Position Trajectory")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.legend()
plt.tight_layout()
plt.show()


df1['time_sec'] = df1['time'] * 1e-9 if df1['time'].max() > 1e12 else df1['time']
plt.figure(figsize=(12, 7))
# X Velocity
plt.plot(df1['time_sec'], df1['ee_vel_field.linear.x'], label='Actual X', color='blue')
plt.plot(df1['time_sec'], df1['twist_cmd_field.linear.x'], '--', label='Desired X', color='blue')

# Y Velocity
plt.plot(df1['time_sec'], df1['ee_vel_field.linear.y'], label='Actual Y', color='green')
plt.plot(df1['time_sec'], df1['twist_cmd_field.linear.y'], '--', label='Desired Y', color='green')

# Z Velocity
plt.plot(df1['time_sec'], df1['ee_vel_field.linear.z'], label='Actual Z', color='red')
plt.plot(df1['time_sec'], df1['twist_cmd_field.linear.z'], '--', label='Desired Z', color='red')

# Labels and legend
plt.title("End-Effector Velocity vs Desired Velocity (XYZ)")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
