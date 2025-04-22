import os
import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
import pyLasaDataset as lasa  # ensure pyLasaDataset is installed

# Determine package root directory (assumes this file lives in <pkg>/utils)
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))


def _process_bag(path):
    """ Process .mat files that is converted from .bag files """
    data_ = loadmat(path)
    data_ = data_['data_ee_pose']
    L = data_.shape[1]

    p_raw = []
    q_raw = []
    t_raw = []

    sample_step = 5
    vel_thresh = 1e-3

    for l in range(L):
        data_l = data_[0, l]['pose'][0, 0]
        pos_traj = data_l[:3, ::sample_step]
        quat_traj = data_l[3:7, ::sample_step]
        time_traj = data_l[-1, ::sample_step].reshape(1, -1)

        # Trim start/end stationary segments
        diffs = np.diff(pos_traj, axis=1)
        vel_mag = np.linalg.norm(diffs, axis=0)
        i0 = np.argmax(vel_mag > vel_thresh)
        i1 = len(vel_mag) - 1 - np.argmax(vel_mag[::-1] > vel_thresh)
        if i0 >= i1:
            raise ValueError("Velocity threshold yields empty segment")

        pos_traj = pos_traj[:, i0:i1]
        quat_traj = quat_traj[:, i0:i1]
        time_traj = time_traj[:, i0:i1]

        p_raw.append(pos_traj.T)
        q_raw.append([R.from_quat(quat_traj[:, i]) for i in range(quat_traj.shape[1])])
        t_raw.append(time_traj.flatten())

    # Compute average dt from first trajectory
    dt = np.mean(np.diff(t_raw[0]))
    return p_raw, q_raw, t_raw, dt


def _get_sequence(seq_file):
    with open(seq_file, 'r') as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_clfd_dataset(task_id=1, num_traj=1, sub_sample=3):
    """
    Load data from CLFD dataset in <pkg_root>/dataset/clfd
    """
    # Paths
    clfd_dir = os.path.join(PKG_ROOT, 'dataset', 'clfd')
    seq_file = os.path.join(clfd_dir, 'robottasks_pos_ori_sequence_4.txt')
    files = _get_sequence(seq_file)
    # pick file
    datafile = os.path.join(clfd_dir, files[task_id])

    data = np.load(datafile)[:, ::sub_sample, :]

    p_raw, q_raw, t_raw = [], [], []
    T = 10.0

    for l in range(num_traj):
        arr = data[l, :, :]
        M = arr.shape[0]
        # convert scalar-first (qw,qx,qy,qz) to [x,y,z,w]
        w = arr[:, 3].copy()
        xyz = arr[:, 4:].copy()
        quats = np.hstack((xyz, w.reshape(-1,1)))

        p_raw.append(arr[:, :3])
        q_raw.append([R.from_quat(q) for q in quats])
        t_raw.append(np.linspace(0, T, M, endpoint=False))

    dt = np.mean(np.diff(t_raw[0]))
    return p_raw, q_raw, t_raw, dt


def load_demo_dataset():
    """
    Load demo data recorded via kinesthetic teaching
    """
    demo_path = os.path.join(PKG_ROOT, 'dataset', 'demo', 'all.mat')
    return _process_bag(demo_path)


def load_UMI():
    """
    Load UMI trajectory from <pkg_root>/dataset/UMI/traj1.npy
    """
    fim = os.path.join(PKG_ROOT, 'dataset', 'UMI', 'traj1.npy')
    traj = np.load(fim)

    # Extract positions and orientations
    p_raw = [traj[i, :3, 3] for i in range(traj.shape[0])]
    rot_mats = [traj[i, :3, :3] for i in range(traj.shape[0])]
    q_raw = [R.from_matrix(m) for m in rot_mats]

    T = 5.0
    dt = T / traj.shape[0]
    t_raw = [np.linspace(0, T, traj.shape[0])]

    return [np.vstack(p_raw)], [q_raw], t_raw, dt


def load_lasa(shape='Sshape'):
    """
    Load LASA handwriting dataset via pyLasaDataset
    """
    try:
        dataset = getattr(lasa.DataSet, shape)
    except AttributeError:
        raise ValueError(f"Unknown LASA dataset '{shape}'")

    dt = dataset.dt
    demo = dataset.demos[0]

    pos2d = demo.pos      # shape (2, N)
    t_vec = np.array(demo.t).flatten()

    # pad to 3D and transpose -> (N,3)
    zeros = np.zeros((1, pos2d.shape[1]))
    p3 = np.vstack((pos2d, zeros)).T

    # identity quaternion [x,y,z,w]
    N = pos2d.shape[1]
    q_id = np.tile([0,0,0,1], (N,1))
    q_raw = [R.from_quat(q_id)]

    return [p3], q_raw, [t_vec], dt
