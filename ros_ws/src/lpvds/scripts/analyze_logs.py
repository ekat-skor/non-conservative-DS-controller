import json
import numpy as np
import matplotlib.pyplot as plt

# Load the logged data
with open('lpvds_run_log.json', 'r') as f:
    log_data = json.load(f)

# Extract arrays
times = np.array([entry['time'] for entry in log_data])
pos_errors = np.array([entry['p_error'] for entry in log_data])
ori_errors = np.array([entry['q_error'] for entry in log_data])
v_cmds = np.array([entry['v_cmd'] for entry in log_data])
w_cmds = np.array([entry['w_cmd'] for entry in log_data])
positions = np.array([entry['p_actual'] for entry in log_data])

# --- Plot 1: Position Error over Time ---
plt.figure()
plt.plot(times - times[0], pos_errors)
plt.title('Position Error Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Position Error (m)')
plt.grid(True)

# --- Plot 2: Orientation Error over Time ---
plt.figure()
plt.plot(times - times[0], ori_errors)
plt.title('Orientation Error Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Orientation Error (rad)')
plt.grid(True)

# --- Plot 3: Velocity Commands over Time ---
plt.figure()
plt.plot(times - times[0], v_cmds)
plt.title('Linear Velocity Commands')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend(['vx', 'vy', 'vz'])
plt.grid(True)

plt.figure()
plt.plot(times - times[0], w_cmds)
plt.title('Angular Velocity Commands')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend(['wx', 'wy', 'wz'])
plt.grid(True)

# --- Plot 4: Trajectory (Baseline vs. Actual) ---
# You may need to add baseline (desired) trajectory points to your logging if you want this more detailed.
p_actual = positions
p_att = np.array(log_data[-1]['p_actual'])  # Assuming the goal is constant

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(p_actual[:, 0], p_actual[:, 1],
        p_actual[:, 2], label='Actual Trajectory')
ax.scatter(p_att[0], p_att[1], p_att[2], c='r', marker='x', label='Attractor')
ax.set_title('3D Trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
