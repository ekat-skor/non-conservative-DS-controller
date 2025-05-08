import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def compute_rmse(desired_velocities, actual_velocities):
    """
    Compute the Root Mean Square Error (RMSE) between desired and actual velocities.

    Parameters:
    - desired_velocities (np.ndarray): An (N x D) array of desired velocity vectors.
    - actual_velocities (np.ndarray): An (N x D) array of actual velocity vectors.

    Returns:
    - float: The RMSE value.
    """
    desired_velocities = np.array(desired_velocities)
    actual_velocities = np.array(actual_velocities)

    if desired_velocities.shape != actual_velocities.shape:
        raise ValueError("Input arrays must have the same shape.")

    squared_error = np.sum((desired_velocities - actual_velocities) ** 2, axis=1)
    mse = np.mean(squared_error)
    rmse = np.sqrt(mse)
    return rmse


def dtw_distance(A, B, window=None):
    N, M = len(A), len(B)
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0

    for i in range(1, N + 1):
        j_start = 1 if window is None else max(1, i - window)
        j_end = M + 1 if window is None else min(M + 1, i + window + 1)
        for j in range(j_start, j_end):
            cost = np.linalg.norm(A[i - 1] - B[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    return D[N, M]
# Load the merged CSV for controller 1
df1 = pd.read_csv('/workspace/ros_ws/conservative_panda_ee_full_20250507_175443.csv')
# control_metrics_1 = pd.read_csv('/workspace/ros_ws/conservative_panda_ee_full_20250505_233910.csv')

# baseline = pdad_csv('/workspace/ros_ws/conservative_panda_ee_full_20250505_233910.csv')


## Load csv for baseline
# df2 = pd.read_csv('/workspace/ros_ws/nc_panda_ee_full_20250505_233910.csv')

##load csv for 

# Extract x, y, z position columns from controller 1
x_real_1 = df1['ee_pose_field.position.x']
y_real_1 = df1['ee_pose_field.position.y']
z_real_1 = df1['ee_pose_field.position.z']

# x_base = baseline['ee_pose_field.position.x']
# y_base = baseline['ee_pose_field.position.y']
# z_base = baseline['ee_pose_field.position.z']

# x_real_2 = df2['ee_pose_field.position.x']
# y_real_2 = df2['ee_pose_field.position.y']
# z_real_2 = df2['ee_pose_field.position.z']

# Create 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_real_1, y_real_1, z_real_1, label='Conservative Controller End-Effector Trajectory', color='blue')
# ax.plot(x_real_2, y_real_2, z_real_2, label = 'non-Conservative Controller End-Effector Trajectory', color = 'red')
# ax.plot(x_base, y_base, z_base, label='basline LPVDS End-Effector Trajectory', color='green')

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

##calculating DTDW:


# ================================
# Uncomment this section once the baseline CSV is available
# ================================

# # Load baseline CSV
# df_baseline = pd.read_csv('/workspace/ros_ws/nc_panda_ee_full_20250505_233910.csv')

# # --- DTWD Calculation ---
# pos_ctrl = df1[['ee_pose_field.position.x', 'ee_pose_field.position.y', 'ee_pose_field.position.z']].to_numpy()
# pos_base = df_baseline[['ee_pose_field.position.x', 'ee_pose_field.position.y', 'ee_pose_field.position.z']].to_numpy()

# dtwd_value = dtw_distance(pos_ctrl, pos_base, window=10)
# print(f"DTWD between controller and baseline trajectories: {dtwd_value:.4f}")

# --- RMSE Calculation ---
vel_actual = df1[['ee_vel_field.linear.x', 'ee_vel_field.linear.y', 'ee_vel_field.linear.z']].to_numpy()
vel_desired = df1[['twist_cmd_field.linear.x', 'twist_cmd_field.linear.y', 'twist_cmd_field.linear.z']].to_numpy()

velocity_rmse = compute_rmse(vel_desired, vel_actual)
print(f"RMSE of end-effector velocity for : {velocity_rmse:.4f}")


