import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Load CSVs ===
baseline_path = '/home/tianyu/ncds_ws/non-conservative-DS-controller/ros_ws/NC_CSVS/baseline.csv'  # <- change to your actual file
experimental_path = '/home/tianyu/ncds_ws/non-conservative-DS-controller/ros_ws/NC_CSVS/non-conservative_panda_ee_full_.csv'  # <- change to your actual file

baseline_df = pd.read_csv(baseline_path)
experimental_df = pd.read_csv(experimental_path)

# === Extract Positions ===
# For baseline (assumed columns: x, y, z)
baseline_xyz = baseline_df[['x', 'y', 'z']].values

# For experimental (eepose_field.position.x, .y, .z)
exp_xyz = experimental_df[
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

# Plot experimental trajectory
ax.plot(exp_xyz[:,0], exp_xyz[:,1], exp_xyz[:,2], label='Experimental Trajectory', linestyle='-')
# Mark start and end points for experimental
ax.scatter(exp_xyz[0,0], exp_xyz[0,1], exp_xyz[0,2], color='lime', marker='o', s=50, label='Experimental Start')
ax.scatter(exp_xyz[-1,0], exp_xyz[-1,1], exp_xyz[-1,2], color='darkred', marker='x', s=50, label='Experimental End')

# Labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Baseline vs Experimental Trajectory')
ax.legend()

plt.tight_layout()
plt.show()
