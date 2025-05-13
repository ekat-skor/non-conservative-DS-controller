import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Load CSVs ===
baseline_path = '/home/tianyu/ncds_ws/non-conservative-DS-controller/ros_ws/C_CSVS/baseline.csv'
conservative_path = '/home/tianyu/ncds_ws/non-conservative-DS-controller/ros_ws/C_CSVS/conservative_panda_ee_full_.csv'
nonconservative_path = '/home/tianyu/ncds_ws/non-conservative-DS-controller/ros_ws/C_CSVS/non-conservative_panda_ee_full_.csv'

baseline_df = pd.read_csv(baseline_path)
conservative_df = pd.read_csv(conservative_path)
nonconservative_df = pd.read_csv(nonconservative_path)

# === Extract Positions ===
# For baseline (assumed columns: x, y, z)
baseline_xyz = baseline_df[['x', 'y', 'z']].values

# For conservative (eepose_field.position.x, .y, .z)
conservative_xyz = conservative_df[
    ['eepose_field.position.x', 'eepose_field.position.y', 'eepose_field.position.z']
].values

# For non-conservative (eepose_field.position.x, .y, .z)
nonconservative_xyz = nonconservative_df[
    ['eepose_field.position.x', 'eepose_field.position.y', 'eepose_field.position.z']
].values

# === Plot ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot baseline trajectory
ax.plot(baseline_xyz[:,0], baseline_xyz[:,1], baseline_xyz[:,2], label='Baseline Trajectory', linestyle='--')
# Mark start and end points for baseline
ax.scatter(baseline_xyz[0,0], baseline_xyz[0,1], baseline_xyz[0,2], color='green', marker='o', s=50, label='Baseline Start')
ax.scatter(baseline_xyz[-1,0], baseline_xyz[-1,1], baseline_xyz[-1,2], color='red', marker='x', s=50, label='Baseline End')

# Plot conservative trajectory
ax.plot(conservative_xyz[:,0], conservative_xyz[:,1], conservative_xyz[:,2], label='Conservative Trajectory', linestyle='-')
# Mark start and end points for conservative
ax.scatter(conservative_xyz[0,0], conservative_xyz[0,1], conservative_xyz[0,2], color='cyan', marker='o', s=50, label='Conservative Start')
ax.scatter(conservative_xyz[-1,0], conservative_xyz[-1,1], conservative_xyz[-1,2], color='blue', marker='x', s=50, label='Conservative End')

# Plot non-conservative trajectory
ax.plot(nonconservative_xyz[:,0], nonconservative_xyz[:,1], nonconservative_xyz[:,2], label='Non-Conservative Trajectory', linestyle='-.')
# Mark start and end points for non-conservative
ax.scatter(nonconservative_xyz[0,0], nonconservative_xyz[0,1], nonconservative_xyz[0,2], color='lime', marker='o', s=50, label='Non-Conservative Start')
ax.scatter(nonconservative_xyz[-1,0], nonconservative_xyz[-1,1], nonconservative_xyz[-1,2], color='darkred', marker='x', s=50, label='Non-Conservative End')

# Labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Baseline vs Conservative vs Non-Conservative Trajectories')
ax.legend()

plt.tight_layout()
plt.show()
