import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Load CSV (update to your actual filename)
fn = '/home/tianyu/ncds_ws/non-conservative-DS-controller/ros_ws/C_CSVS/conservative_panda_ee_full_.csv'
df = pd.read_csv(fn)

# 2) Normalize time into seconds
if df['time'].max() > 1e12:
    df['t_s'] = df['time'] * 1e-9
else:
    df['t_s'] = df['time']
t = df['t_s']
df['t_s'] -= df['t_s'].iloc[0]

# 3) Identify the columns we need

# EE linear velocity
vel_cols = sorted(
    [c for c in df.columns if c.startswith('eevel_field.linear')],
    key=lambda s: s.split('.')[-1]  # ensure x,y,z order
)

# External torque
tau_cols = sorted(
    [c for c in df.columns if c.startswith('fext_field.wrench.torque')],
    key=lambda s: s.split('.')[-1]
)

# Full Dmat entries
d_cols = [f'Dmat_field.data{i}' for i in range(9)]
missing = [c for c in d_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing Dmat columns: {missing}")

# 4) Vectorized extraction
V   = df[vel_cols].values           # shape (N,3)
Tau = df[tau_cols].values           # shape (N,3)
D   = df[d_cols].values.reshape(-1,3,3)  # shape (N,3,3)

# 5) Compute conservative Wdot:
#    Wdot = - v^T D v  +  v^T τ

quad = np.einsum('ni,nij,nj->n', V, D, V)   # vᵀ D v
dot  = np.einsum('ni,ni->n',      V, Tau)  # v · τ

df['wdot_conservative'] = -quad + dot

# 6) Plot only the conservative Wdot
plt.figure(figsize=(8,4))
plt.plot(t, df['wdot_conservative'], label=r'$\dot W_{cons}$')
plt.xlabel('Time [s]')
plt.ylabel(r'$\dot W$')
plt.title('Conservative DS Power $\dot W$ over Time')
plt.grid(True)
plt.tight_layout()
plt.show()

# 7) Save the new CSV
out = fn.replace('.csv', '_cons_wdot.csv')
df.to_csv(out, index=False)
