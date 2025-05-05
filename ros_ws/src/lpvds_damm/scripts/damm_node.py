#!/usr/bin/env python
import rospy
import json
import numpy as np
from geometry_msgs.msg import Pose, Twist
from scipy.spatial.transform import Rotation as R
import os
import sys
from std_msgs.msg import Float64

sys.path.insert(
    0, '/workspace/ros_ws/src/lpvds_damm/src/util/'
)

# Utility functions for quaternion operations
from quat_tools import riem_log, riem_exp, parallel_transport

class DAMMLPVDSNode(object):
    def __init__(self):
        rospy.init_node('damm_lpvds_node')

        # PUBLISH twist to controller 
        self.cmd_pub = rospy.Publisher(
            '/passiveDS/desired_twist', Twist, queue_size=1)
    
        # PUBLISH the negative gradient of the lyapunov function (conservative_des_vel)
        self.conservative_des_vel_pub = rospy.Publisher(
            '/passiveDS/conservative_des_vel', Twist, queue_size=1)
        
        # SUBSCRIBE to actual end-effector pose
        self.current_pose = None
        rospy.Subscriber(
            '/franka_state_controller/ee_pose', Pose,
            self._pose_callback, queue_size=1)

        # Load the learned matrix P from p.json
        self.P = self.load_P_matrix()

        # Load position and orientation models from parameters
        position_model_path = rospy.get_param('~position_model_json', '/path/to/j_shape_position.json')
        orientation_model_path = rospy.get_param('~orientation_model_json', '/path/to/j_shape_orientation.json')
        
        self.load_trained_position_lpvds(position_model_path)
        self.load_trained_orientation_lpvds(orientation_model_path)

        # Timer to periodically call the update function (100 Hz)
        self.timer = rospy.Timer(rospy.Duration(0.01), self.timer_cb)

    def _pose_callback(self, msg: Pose):
        self.current_pose = msg


    # for Lyapunov function calculation : V = [(x-x*)^T]P[(x-x*)]
    def load_P_matrix(self):
        """Load the matrix P from the p.json file."""
        p_file_path = rospy.get_param('~p_json_file', '/workspace/ros_ws/src/lpvds_damm/trained_ds/p.json')  # Hardcoded for testing
        rospy.loginfo(f"Attempting to load P matrix from: {p_file_path}")
        try:
            with open(p_file_path, 'r') as f:
                P_data = json.load(f)
                P = np.array(P_data)  # Convert JSON list back to a NumPy array
                if P is None or len(P) == 0:
                    rospy.logerr("Loaded P matrix is empty or None!")
                    return None
                rospy.loginfo(f"Successfully loaded P matrix from p.json with shape: {P.shape}")
                return P
        except FileNotFoundError:
            rospy.logerr(f"p.json file not found at {p_file_path}")
            return None



    def compute_lyapunov_function(self, x):
        """Compute the Lyapunov function V(x) = (x - x*)^T P (x - x*)"""
        if self.P is None:
            rospy.logerr("Matrix P is not loaded")
            return None
        
        # Shift position by subtracting attractor
        x_sh = x - self.x_att
        V = np.dot(x_sh.T, np.dot(self.P, x_sh))  # Lyapunov function
        return V

    def compute_gradient_lyapunov(self, x):
        """Compute the gradient of the Lyapunov function V(x) = (x - x*)^T P (x - x*)"""
        if self.P is None:
            rospy.logerr("Matrix P is not loaded")
            return None
        
        # Shift position by subtracting attractor-- is this actually the 2D gradient? check 
        x_sh = x - self.x_att
        gradient = 2 * np.dot(self.P, x_sh)  # Gradient of V(x) with respect to x 
        return gradient

    def publish_conservative_des_vel(self, gradient):
        """Publish the negative gradient of the Lyapunov function"""
        if gradient is None:
            rospy.logwarn("Gradient is None, skipping publication of conservative_des_vel.")
            return
        conservative_des_vel = -gradient  # Negative gradient of Lyapunov function
        twist_msg = Twist()
        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = conservative_des_vel
        self.conservative_des_vel_pub.publish(twist_msg)

    def load_trained_position_lpvds(self, json_path: str):
        """
        Read position-related JSON and pull out learned parameters.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract position-related parameters
        self.dt = data.get('dt', 0.01)  # Default to 0.01 if 'dt' is missing
        self.K_pos = data['K']
        self.priors_pos = np.array(data['Prior'])

        Mu_pos = np.array(data['Mu'])
        C_pos = self.priors_pos.size
        D_pos = Mu_pos.size // C_pos
        Mu_pos = Mu_pos.reshape((C_pos, D_pos))

        self.mu_pos = Mu_pos
        Sigma_pos = np.array(data['Sigma'])
        Sdim_pos = int(np.sqrt(Sigma_pos.size / C_pos))
        self.sigmas_pos = Sigma_pos.reshape((C_pos, Sdim_pos, Sdim_pos))

        self.A_pos = np.array(data['A']).reshape((C_pos, 3, 3))
        
        # Set attractor and assign it to att_pos
        self.x_att = np.array(data['attractor'])
        self.att_pos = self.x_att  

        rospy.loginfo(f"Loaded DAMM-LPVDS position model with K={self.K_pos}, dt={self.dt}")
        rospy.loginfo(f"Attractor position (att_pos): {self.att_pos}")


    def load_trained_orientation_lpvds(self, json_path: str):
        """
        Read orientation-related JSON and pull out learned parameters.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract orientation-related parameters
        self.K_ori = data['K']
        self.priors_ori = np.array(data['Prior'])

        Mu_ori = np.array(data['Mu'])
        C_ori = self.priors_ori.size
        D_ori = Mu_ori.size // C_ori
        Mu_ori = Mu_ori.reshape((C_ori, D_ori))

        self.mu_ori = Mu_ori
        Sigma_ori = np.array(data['Sigma'])
        Sdim_ori = int(np.sqrt(Sigma_ori.size / C_ori))
        self.sigmas_ori = Sigma_ori.reshape((C_ori, Sdim_ori, Sdim_ori))

        self.A_ori = np.array(data['A_ori']).reshape((C_ori, 4, 4))
        self.att_ori = R.from_quat(data['att_ori'])

        rospy.loginfo(f"Loaded Quaternion-DS orientation model with K={self.K_ori}")
        
    def compute_logprob(self, x, pis, means, covs):
        """
        Vectorized log-posterior probability for a Gaussian mixture.
        """
        K, D = means.shape
        if covs.shape[1] != D:
            covs = covs[:, :D, :D]

        cov_inv = np.linalg.inv(covs)
        cov_det = np.linalg.det(covs)

        # Convert rotations (if they are Rotation objects) to quaternions or numpy arrays
        if isinstance(x, R):
            x = x.as_quat()  # Convert rotation object to quaternion
        if isinstance(means, R):
            means = means.as_quat()  # Convert rotation object to quaternion

        # Ensure x and means are 2D for broadcasting
        x = np.atleast_2d(x)  # Ensure x is 2D (1 row, multiple columns)
        means = np.atleast_2d(means)  # Ensure means is 2D (1 row, multiple columns)

        centered_x = x[:, np.newaxis, :] - means[:, np.newaxis, :]
        exponent = -0.5 * np.einsum('kni,kij,knj->kn', centered_x, cov_inv, centered_x)

        log_coeff = -0.5 * (D * np.log(2 * np.pi) + np.log(cov_det)).reshape(-1, 1)
        log_pdf = log_coeff + exponent
        log_pis = np.log(pis).reshape(-1, 1)
        logProb = log_pis + log_pdf

        maxPost = np.max(logProb, axis=0, keepdims=True)
        expPost = np.exp(logProb - maxPost)
        postProb = expPost / np.sum(expPost, axis=0, keepdims=True)

        return postProb
  

    def step_linear(self, curr_x, Priors, Mu, Sigma, As, goal):
        """
        Linear component of LPV-DS model for position.
        """
        linear_vel = np.zeros(3)
        Sigma_pos = Sigma[:, :3, :3]
        gamma = self.compute_logprob(curr_x.reshape(1, -1), Priors, Mu, Sigma_pos)
        for k in range(len(gamma)):
            linear_vel += (gamma[k] * (As[k] @ (curr_x - goal))).flatten()
        return linear_vel

    def step_angular(self, q_in, K, priors, mus, sigmas, q_att, A_ori, dt):
        """
        Angular component of LPV-DS model for orientation.
        """
        gamma = self.compute_logprob(q_in, priors, mus, sigmas)
        q_diff = riem_log(q_att, q_in)
        q_out_att = sum(gamma[k, 0] * (A_ori[k] @ q_diff.T) for k in range(K))
        q_out_body = parallel_transport(q_att, q_in, q_out_att.T)
        q_out_q = riem_exp(q_in, q_out_body)
        q_out = R.from_quat(q_out_q.reshape(4,))
        omega = self.compute_ang_vel(q_in, q_out, dt)

        return omega

    def compute_ang_vel(self, q_k, q_kp1, dt=0.01):
        """
        Compute angular velocity from two quaternions.
        """
        dq = q_kp1 * q_k.inv()
        rotvec = dq.as_rotvec()
        return rotvec / dt

    def timer_cb(self, event):
        if self.current_pose is None:
            return

        pos = self.current_pose.position
        x = np.array([pos.x, pos.y, pos.z])

        # Compute the gradient of the Lyapunov function
        gradient = self.compute_gradient_lyapunov(x)

        # Publish the negative gradient as conservative_des_vel
        self.publish_conservative_des_vel(gradient)

        # Linear position-based LPV-DS
        lin_vel = self.step_linear(x, self.priors_pos, self.mu_pos, self.sigmas_pos, self.A_pos, self.att_pos)

        # Angular orientation-based LPV-DS
        ori = self.current_pose.orientation
        q = R.from_quat([ori.x, ori.y, ori.z, ori.w])
        ang_vel = self.step_angular(q, self.K_ori, self.priors_ori, self.mu_ori, self.sigmas_ori, self.att_ori, self.A_ori, self.dt)

        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.linear.z = lin_vel
        cmd.angular.x, cmd.angular.y, cmd.angular.z = ang_vel # need to implement tank for angular velocity as well 
        # cmd.angular.x = 0.0
        # cmd.angular.y = 0.0
        # cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

if __name__ == '__main__':
    node = DAMMLPVDSNode()
    rospy.spin()








####### Yuka-Kat version functioning damm_node without conservative_vel ###########################

# #!/usr/bin/env python
# import rospy
# import json
# import numpy as np
# from geometry_msgs.msg import Pose, Twist
# from scipy.spatial.transform import Rotation as R
# import os
# import sys

# sys.path.insert(
#     0, '/workspace/ros_ws/src/lpvds_damm/src/util/'
# )

# # Utility functions for quaternion operations
# from quat_tools import riem_log, riem_exp, parallel_transport

# class DAMMLPVDSNode(object):
#     def __init__(self):
#         rospy.init_node('damm_lpvds_node')

#         # PUBLISH twist to controller 
#         self.cmd_pub = rospy.Publisher(
#             '/passiveDS/desired_twist', Twist, queue_size=1)


#         # SUBSCRIBE to actual end-effector pose
#         self.current_pose = None
#         rospy.Subscriber(
#             '/franka_state_controller/ee_pose', Pose,
#             self._pose_callback, queue_size=1)

#         # Load position and orientation models from parameters
#         position_model_path = rospy.get_param('~position_model_json', '/path/to/j_shape_position.json')
#         orientation_model_path = rospy.get_param('~orientation_model_json', '/path/to/j_shape_orientation.json')
        
#         self.load_trained_position_lpvds(position_model_path)
#         self.load_trained_orientation_lpvds(orientation_model_path)

#         # Timer to periodically call the update function (100 Hz)
#         self.timer = rospy.Timer(rospy.Duration(0.01), self.timer_cb)

#     def _pose_callback(self, msg: Pose):
#         self.current_pose = msg

#     def load_trained_position_lpvds(self, json_path: str):
#         """
#         Read position-related JSON and pull out learned parameters.
#         """
#         with open(json_path, 'r') as f:
#             data = json.load(f)

#         # Extract position-related parameters
#         self.dt = data.get('dt', 0.01)  # Default to 0.01 if 'dt' is missing
#         self.K_pos = data['K']
#         self.priors_pos = np.array(data['Prior'])

#         Mu_pos = np.array(data['Mu'])
#         C_pos = self.priors_pos.size
#         D_pos = Mu_pos.size // C_pos
#         Mu_pos = Mu_pos.reshape((C_pos, D_pos))

#         self.mu_pos = Mu_pos
#         Sigma_pos = np.array(data['Sigma'])
#         Sdim_pos = int(np.sqrt(Sigma_pos.size / C_pos))
#         self.sigmas_pos = Sigma_pos.reshape((C_pos, Sdim_pos, Sdim_pos))

#         self.A_pos = np.array(data['A']).reshape((C_pos, 3, 3))
#         self.att_pos = np.array(data['attractor'])

#         rospy.loginfo(f"Loaded DAMM-LPVDS position model with K={self.K_pos}, dt={self.dt}")

#     def load_trained_orientation_lpvds(self, json_path: str):
#         """
#         Read orientation-related JSON and pull out learned parameters.
#         """
#         with open(json_path, 'r') as f:
#             data = json.load(f)

#         # Extract orientation-related parameters
#         self.K_ori = data['K']
#         self.priors_ori = np.array(data['Prior'])

#         Mu_ori = np.array(data['Mu'])
#         C_ori = self.priors_ori.size
#         D_ori = Mu_ori.size // C_ori
#         Mu_ori = Mu_ori.reshape((C_ori, D_ori))

#         self.mu_ori = Mu_ori
#         Sigma_ori = np.array(data['Sigma'])
#         Sdim_ori = int(np.sqrt(Sigma_ori.size / C_ori))
#         self.sigmas_ori = Sigma_ori.reshape((C_ori, Sdim_ori, Sdim_ori))

#         self.A_ori = np.array(data['A_ori']).reshape((C_ori, 4, 4))
#         self.att_ori = R.from_quat(data['att_ori'])

#         rospy.loginfo(f"Loaded Quaternion-DS orientation model with K={self.K_ori}")

    
    
#     def compute_logprob(self, x, pis, means, covs):
#         """
#         Vectorized log-posterior probability for a Gaussian mixture.
#         """
#         K, D = means.shape
#         if covs.shape[1] != D:
#             covs = covs[:, :D, :D]

#         cov_inv = np.linalg.inv(covs)
#         cov_det = np.linalg.det(covs)

#         # Convert rotations (if they are Rotation objects) to quaternions or numpy arrays
#         if isinstance(x, R):
#             x = x.as_quat()  # Convert rotation object to quaternion
#         if isinstance(means, R):
#             means = means.as_quat()  # Convert rotation object to quaternion

#         # Ensure x and means are 2D for broadcasting
#         x = np.atleast_2d(x)  # Ensure x is 2D (1 row, multiple columns)
#         means = np.atleast_2d(means)  # Ensure means is 2D (1 row, multiple columns)

#         centered_x = x[:, np.newaxis, :] - means[:, np.newaxis, :]
#         exponent = -0.5 * np.einsum('kni,kij,knj->kn', centered_x, cov_inv, centered_x)

#         log_coeff = -0.5 * (D * np.log(2 * np.pi) + np.log(cov_det)).reshape(-1, 1)
#         log_pdf = log_coeff + exponent
#         log_pis = np.log(pis).reshape(-1, 1)
#         logProb = log_pis + log_pdf

#         maxPost = np.max(logProb, axis=0, keepdims=True)
#         expPost = np.exp(logProb - maxPost)
#         postProb = expPost / np.sum(expPost, axis=0, keepdims=True)

#         return postProb


#     def step_linear(self, curr_x, Priors, Mu, Sigma, As, goal):
#         """
#         Linear component of LPV-DS model for position.
#         """
#         linear_vel = np.zeros(3)
#         Sigma_pos = Sigma[:, :3, :3]
#         gamma = self.compute_logprob(curr_x.reshape(1, -1), Priors, Mu, Sigma_pos)
#         for k in range(len(gamma)):
#             linear_vel += (gamma[k] * (As[k] @ (curr_x - goal))).flatten()
#         return linear_vel

#     def step_angular(self, q_in, K, priors, mus, sigmas, q_att, A_ori, dt):
#         """
#         Angular component of LPV-DS model for orientation.
#         """
#         gamma = self.compute_logprob(q_in, priors, mus, sigmas)
#         q_diff = riem_log(q_att, q_in)
#         q_out_att = sum(gamma[k, 0] * (A_ori[k] @ q_diff.T) for k in range(K))
#         q_out_body = parallel_transport(q_att, q_in, q_out_att.T)
#         q_out_q = riem_exp(q_in, q_out_body)
#         q_out = R.from_quat(q_out_q.reshape(4,))
#         omega = self.compute_ang_vel(q_in, q_out, dt)

#         return omega

#     def compute_ang_vel(self, q_k, q_kp1, dt=0.01):
#         """
#         Compute angular velocity from two quaternions.
#         """
#         dq = q_kp1 * q_k.inv()
#         rotvec = dq.as_rotvec()
#         return rotvec / dt

#     def timer_cb(self, event):
#         if self.current_pose is None:
#             return

#         pos = self.current_pose.position
#         x = np.array([pos.x, pos.y, pos.z])

#         ori = self.current_pose.orientation
#         q = R.from_quat([ori.x, ori.y, ori.z, ori.w])

#         # Linear position-based LPV-DS
#         lin_vel = self.step_linear(x, self.priors_pos, self.mu_pos, self.sigmas_pos, self.A_pos, self.att_pos)

#         # Angular orientation-based LPV-DS
#         ang_vel = self.step_angular(q, self.K_ori, self.priors_ori, self.mu_ori, self.sigmas_ori, self.att_ori, self.A_ori, self.dt)

#         cmd = Twist()
#         cmd.linear.x, cmd.linear.y, cmd.linear.z = lin_vel
#         cmd.angular.x, cmd.angular.y, cmd.angular.z = ang_vel
#         self.cmd_pub.publish(cmd)

# if __name__ == '__main__':
#     node = DAMMLPVDSNode()
#     rospy.spin()