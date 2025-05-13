import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# 1) Load CSV
fn = '/home/tianyu/ncds_ws/non-conservative-DS-controller/ros_ws/NC_CSVS/non-conservative_panda_ee_full_.csv'
df = pd.read_csv(fn)

# 2) Normalize time
if df['time'].max() > 1e12:
    df['t_s'] = df['time'] * 1e-9
else:
    df['t_s'] = df['time']
df['t_s'] -= df['t_s'].iloc[0]
t = df['t_s']

# 3) Pick out the right columns

# EE velocity: look for eevel_field.linear.[x|y|z]
vel_cols = sorted(
    [c for c in df.columns if re.match(r'eevel_field\.linear\.[xyz]$', c)],
    key=lambda s: s[-1]
)
if len(vel_cols) != 3:
    raise RuntimeError(f"Expected 3 EE‐velocity columns, got {len(vel_cols)}: {vel_cols}")

# External torque: fext_field.wrench.torque.[x|y|z]
tau_cols = sorted(
    [c for c in df.columns if re.match(r'fext_field\.wrench\.torque\.[xyz]$', c)],
    key=lambda s: s[-1]
)
if len(tau_cols) != 3:
    raise RuntimeError(f"Expected 3 torque columns, got {len(tau_cols)}: {tau_cols}")

# Dmat entries: Dmat_field.data0…data8
d_cols = sorted(
    [c for c in df.columns if re.match(r'Dmat_field\.data\d+$', c)],
    key=lambda s: int(s.split('data')[-1])
)
if len(d_cols) != 9:
    raise RuntimeError(f"Expected 9 Dmat entries, got {len(d_cols)}: {d_cols}")

# Scalar params
for name in ['alpha_field.data','betar_field.data','betas_field.data',
             's_field.data','sdot_field.data','z_field.data','eigval0_field.data']:
    if name not in df.columns:
        raise RuntimeError(f"Missing scalar column: {name}")

# 4) Compute wdot
V    = df[vel_cols].values
Tau  = df[tau_cols].values
D    = df[d_cols].values.reshape(-1,3,3)

alpha  = df['alpha_field.data'].values
beta_r = df['betar_field.data'].values
beta_s = df['betas_field.data'].values
z      = df['z_field.data'].values
lam0   = df['eigval0_field.data'].values

quad  = np.einsum('ni,nij,nj->n', V, D, V)
dot   = np.einsum('ni,ni->n', V, Tau)
param = (beta_r - beta_s) * lam0 * z

df['wdot'] = -(1 - alpha) * quad + param + dot


# 5) Plot only wdot
plt.figure(figsize=(8,4))
plt.ylim(-9, 3)
plt.plot(t, df['wdot'])
plt.xlabel('Time [s]')
plt.ylabel('Wdot')
plt.title('Non-Conservative Controller Power $\dot W$ over Time')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6) Save augmented CSV
out_fn = fn.replace('.csv', '_with_wdot.csv')
df.to_csv(out_fn, index=False)
