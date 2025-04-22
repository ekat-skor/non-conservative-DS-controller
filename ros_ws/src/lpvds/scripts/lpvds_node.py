#!/usr/bin/env python
import rospy
import json
import numpy as np
from geometry_msgs.msg import Pose, Twist
from scipy.spatial.transform import Rotation as R
import os, sys

# ensure your package src/ is on the path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(__file__, '..', '..', 'src'))
)

from lpvds.se3_class import se3_class
from lpvds.gmm_class import gmm_class

# -----------------------------------------------------------------------------
# Helper: load trained SE3-LPVDS model from JSON
# -----------------------------------------------------------------------------
def load_trained_se3(json_path: str) -> se3_class:
    """
    Load a previously-trained SE3-LPVDS model from JSON and initialize
    the LPV-DS object (including clustering) for real-time _step().
    Also stores the initial pose from the training data.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    K  = data['K']
    dt = data['dt']

    se3 = se3_class.__new__(se3_class)
    se3.K_init   = K
    se3.dt       = dt
    se3.p_att    = np.array(data['att_pos'])
    se3.q_att    = R.from_quat(data['att_ori'])
    se3.tol      = 1e-2
    se3.max_iter = 5000
    se3.N        = 7  # pos(3)+quat(4)

    se3.A_pos = np.array(data['A_pos']).reshape((2*K, 3, 3))
    se3.A_ori = np.array(data['A_ori']).reshape((2*K, 4, 4))

    # Re-run preprocessing to reconstruct GMM
    from lpvds.util import load_tools, process_tools
    p_raw, q_raw, t_raw, dt_raw = load_tools.load_demo_dataset()
    p_in_proc, q_in_proc, t_in_proc = process_tools.pre_process(
        p_raw, q_raw, t_raw, opt="savgol")
    p_out_proc, q_out_proc = process_tools.compute_output(
        p_in_proc, q_in_proc, t_in_proc)
    p_init_proc, q_init_proc, _, _ = process_tools.extract_state(
        p_in_proc, q_in_proc)

    # Fix: store initial pose as numpy arrays, not lists
    se3.p_init = np.array(p_init_proc[0])
    se3.q_init = q_init_proc[0]
    rospy.loginfo(f"SE3 model initial position: {se3.p_init}")
    rospy.loginfo(f"SE3 model initial orientation (quat): {se3.q_init.as_quat()}")

    p_in_roll, q_in_roll, _, _ = process_tools.rollout_list(
        p_in_proc, q_in_proc, p_out_proc, q_out_proc)
    se3.p_in = p_in_roll
    se3.q_in = q_in_roll
    se3.K_init = K
    se3._cluster()
    return se3

# -----------------------------------------------------------------------------
# ROS Node: uses trained DS to publish desired twists
# -----------------------------------------------------------------------------
class LPVDSNode(object):
    def __init__(self):
        rospy.init_node('lpvds_node')

        # Load trained model
        model_path = rospy.get_param('~model_path',
            os.path.join(os.path.dirname(__file__), '..', 'output.json'))
        rospy.loginfo(f"Loading trained LPVDS model from: {model_path}")
        self.se3 = load_trained_se3(model_path)

        # Publisher for commanded twist
        self.cmd_pub = rospy.Publisher(
            '/passiveDS/desired_twist', Twist, queue_size=1)

        # Publish initial twist to drive to the demo start pose
        p_init = self.se3.p_init
        q_init = self.se3.q_init
        p_in = p_init.reshape(1, -1)
        _, _, _, v_col, w_col = self.se3._step(p_in, q_init, self.se3.dt)
        v_init = v_col.flatten().tolist()
        w_init = w_col.flatten().tolist()
        # cmd_init = Twist()
        # cmd_init.linear.x, cmd_init.linear.y, cmd_init.linear.z = v_init
        # cmd_init.angular.x, cmd_init.angular.y, cmd_init.angular.z = w_init
        # rospy.loginfo("Publishing initial twist to reach start pose...")
        # for _ in range(20):
        #     self.cmd_pub.publish(cmd_init)
        #     rospy.sleep(0.05)
        # rospy.sleep(10.0)
        # rospy.loginfo("Reached init pose")

        # Storage for latest EE pose
        self.current_pose = None
        rospy.Subscriber(
            '/franka_state_controller/ee_pose', Pose,
            self._pose_callback, queue_size=1)

        # Timer @100 Hz
        self.timer = rospy.Timer(
            rospy.Duration(0.01), self.timer_cb)

    def _pose_callback(self, msg: Pose):
        self.current_pose = msg

    def timer_cb(self, event):
        if self.current_pose is None:
            rospy.logwarn_throttle(5.0, "No EE pose received yet")
            return
        p = self.current_pose.position
        o = self.current_pose.orientation
        p_cur = np.array([p.x, p.y, p.z])
        q_cur = R.from_quat([o.x, o.y, o.z, o.w])

        p_in = p_cur.reshape(1, -1)
        _, _, _, v_col, w_col = self.se3._step(p_in, q_cur, self.se3.dt)

        v = v_col.flatten().tolist()
        w = w_col.flatten().tolist()
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.linear.z = v
        cmd.angular.x, cmd.angular.y, cmd.angular.z = w
        self.cmd_pub.publish(cmd)

if __name__ == '__main__':
    node = LPVDSNode()
    rospy.spin()
