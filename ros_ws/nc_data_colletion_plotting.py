import pandas as pd
import matplotlib.pyplot as plt

# ================================
# Load the merged CSV (update path!)
# ================================
df = pd.read_csv('/home/tianyu/ncds_ws/non-conservative-DS-controller/ros_ws/NC_CSVS/non-conservative_panda_ee_full_.csv')

# ================================
# Fix time units
# ================================
df['time_sec'] = df['time'] * 1e-9 if df['time'].max() > 1e12 else df['time']
df['time_sec'] -= df['time_sec'].iloc[0]

# ================================
# Extract data
# ================================
t = df['time_sec']


alpha  = df['alpha_field.data']
beta_r = df['betar_field.data']
beta_s = df['betas_field.data']
s      = df['s_field.data']
sdot   = df['sdot_field.data']
z      = df['z_field.data']

# ================================
# Create stacked plots
# ================================
fig, axes = plt.subplots(6, 1, sharex=True, figsize=(10, 12))
titles = ['alpha', 'beta_r', 'beta_s', 's', 'sdot', 'z']
data = [alpha, beta_r, beta_s, s, sdot, z]

for ax, title, y in zip(axes, titles, data):
    ax.plot(t, y)
    ax.set_ylabel(title)
    ax.grid(True)

    ax.relim()
    ax.autoscale_view(tight=True, scalex=False, scaley=True)



print(f"s  →  min: {s.min():.30f},  max: {s.max():.30f}")
axes[-1].set_xlabel('Time [s]')
fig.suptitle('Non-Conservative Passive DS Parameters Over Time', fontsize=16)
plt.tight_layout()
plt.show()
