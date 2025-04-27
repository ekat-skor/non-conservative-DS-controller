#!/usr/bin/env python
import rospy
import json
import numpy as np
from geometry_msgs.msg import Pose, Twist
from scipy.spatial.transform import Rotation as R
from scipy.stats import multivariate_normal
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(__file__, '..', '..', 'src'))
)

# utility functions for quaternion operations
from lpvds.util.quat_tools import riem_log, riem_exp, parallel_transport


class LPVDSNode(object):
    def __init__(self):
        rospy.init_node('lpvds_node')

        # PUBLISH twist to controller 
        self.cmd_pub = rospy.Publisher(
            '/passiveDS/desired_twist', Twist, queue_size=1)

        # SUBSCRIBE to actual end-effector pose
        self.current_pose = None
        rospy.Subscriber(
            '/franka_state_controller/ee_pose', Pose,
            self._pose_callback, queue_size=1)

        # load all LPV-DS parameters from JSON
        model_path = rospy.get_param('~model_json',
                                     '/path/to/default/SE3-LPVDS.json')
        self.load_trained_lpvds(model_path)

        # 100 Hz
        self.timer = rospy.Timer(rospy.Duration(0.01), self.timer_cb)

    # pose message stored 
    def _pose_callback(self, msg: Pose):
        self.current_pose = msg


    def load_trained_lpvds(self, json_path: str):
        """
        Read JSON and pull out learned parameters.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        # number of mixture components for *each* half of orientation
        self.K  = data['K']               
        self.dt = data['dt']              

        # prior probabilities, shape=(2*K,)
        self.priors = np.array(data['Prior'])

        # Mu = center of Gaussians, C = number of components, D = dim of each component 
        Mu = np.array(data['Mu'])
        C  = self.priors.size            
        D  = Mu.size // C                
        Mu = Mu.reshape((C, D))
        if D == 7:
            # first 3 dims = position means, last 4 = quaternion means
            self.mu_pos = Mu[:, :3]
            self.mu_ori = Mu[:, 3:]
        elif D == 3:
            self.mu_pos = Mu
            self.mu_ori = None
        else:
            raise RuntimeError(f"Unexpected Mu dimension D={D}")

        # covariance: flatten list → (C,D,D)
        Sigma = np.array(data['Sigma'])
        Sdim  = int(np.sqrt(Sigma.size / C))
        self.sigmas = Sigma.reshape((C, Sdim, Sdim))

        # the A matrices for position and orientation
        self.A_pos = np.array(data['A_pos']).reshape((C, 3, 3))
        self.A_ori = np.array(data['A_ori']).reshape((C, 4, 4))

        # attractor (goal) in position and orientation
        self.att_pos = np.array(data['att_pos'])          
        self.att_ori = R.from_quat(data['att_ori'])       

        rospy.loginfo(f"Loaded SE3-LPVDS with K={self.K}, dt={self.dt}")


    # determines log-posterior probabilities for GMM -- used to determine likelihood of current state x for each mixture component 
    def compute_logprob(self, x, pis, means, covs):
        """
        Vectorized log-posterior probability for a Gaussian mixture:
        x: shape (N, D)
        means: shape (K, D)
        covs: shape (K, D', D') where D' >= D
        pis: shape (K,)
        """
        K, D = means.shape
        # trim covariances if they are larger than D
        if covs.shape[1] != D:
            covs = covs[:, :D, :D]

        # invert and det all K covs
        cov_inv = np.linalg.inv(covs)     
        cov_det = np.linalg.det(covs)     

        # center data
        centered_x = x[np.newaxis, :, :] - means[:, np.newaxis, :]

        # exponent term: shape (K, N)
        exponent = -0.5 * np.einsum(
            'kni,kij,knj->kn',
            centered_x, cov_inv, centered_x
        )

        # normalization constant
        log_coeff = -0.5 * (
            D * np.log(2 * np.pi) + np.log(cov_det)
        ).reshape(-1, 1)

        log_pdf = log_coeff + exponent
        log_pis = np.log(pis).reshape(-1, 1)  # (K, 1)
        logProb = log_pis + log_pdf           # (K, N)

        # posterior probabilities
        maxPost = np.max(logProb, axis=0, keepdims=True)
        expPost = np.exp(logProb - maxPost)
        postProb = expPost / np.sum(expPost, axis=0, keepdims=True)

        return postProb


    def step_linear(self, curr_x, Priors, Mu, Sigma, As, goal):
        """
        LPV-DS linear component: returns a 3-vector velocity
        """
        linear_vel = np.zeros(3)
        # use only the 3×3 sub-blocks of Sigma for position
        Sigma_pos = Sigma[:, :3, :3]
        gamma = self.compute_logprob(
            curr_x.reshape(1, -1),
            Priors, Mu, Sigma_pos
        )
        for k in range(len(gamma)):
            linear_vel += (gamma[k] * (As[k] @ (curr_x - goal))).flatten()
        return linear_vel


    def logProb(self, q_in, K, priors, mus, sigmas):
        """
        Compute posterior probabilities for quaternion mixture.
        q_in: scipy Rotation or list thereof
        """
        # ensure covariances match quaternion dim=4
        D = mus.shape[1]
        if sigmas.shape[1] != D:
            sigmas = sigmas[:, -D:, -D:]

        # prepare output shape
        if isinstance(q_in, list):
            logProb = np.zeros((2*K, len(q_in)))
        else:
            logProb = np.zeros((2*K, 1))

        # first K components
        for k in range(K):
            prior_k = priors[k]
            mu_k    = R.from_quat(mus[k])
            normal_k = multivariate_normal(
                np.zeros(D), sigmas[k], allow_singular=True
            )
            q_k = riem_log(mu_k, q_in)
            logProb[k, :] = np.log(prior_k) + normal_k.logpdf(q_k)

        # dual-quaternion half
        for k in range(K):
            prior_k = priors[K+k]
            mu_k    = R.from_quat(mus[K+k])
            normal_k = multivariate_normal(
                np.zeros(D), sigmas[K+k], allow_singular=True
            )
            q_k = riem_log(mu_k, q_in)
            logProb[K+k, :] = np.log(prior_k) + normal_k.logpdf(q_k)

        # posterior normalization
        maxPost = np.max(logProb, axis=0, keepdims=True)
        expPost = np.exp(logProb - maxPost)
        postProb = expPost / np.sum(expPost, axis=0, keepdims=True)

        return postProb


    def compute_ang_vel(self, q_k, q_kp1, dt=0.01):
        """
        Compute angular velocity from two quaternions
        """
        dq = q_kp1 * q_k.inv()
        rotvec = dq.as_rotvec()
        return rotvec / dt


    def step_angular(self, q_in, K, priors, mus, sigmas, q_att, A_ori, dt):
        """
        LPV-DS angular component: returns a 3-vector omega
        """
        gamma = self.logProb(q_in, K, priors, mus, sigmas)

        # primary cover
        q_diff = riem_log(q_att, q_in)
        q_out_att = sum(
            gamma[k, 0] * (A_ori[k] @ q_diff.T)
            for k in range(K)
        )
        q_out_body = parallel_transport(q_att, q_in, q_out_att.T)
        q_out_q    = riem_exp(q_in, q_out_body)
        q_out      = R.from_quat(q_out_q.reshape(4,))
        omega      = self.compute_ang_vel(q_in, q_out, dt)

        # dual cover
        q_att_dual = R.from_quat(-q_att.as_quat())
        q_diff_dual = riem_log(q_att_dual, q_in)
        q_out_att_dual = sum(
            gamma[K+k, 0] * (A_ori[K+k] @ q_diff_dual.T)
            for k in range(K)
        )
        q_out_body_dual = parallel_transport(q_att_dual, q_in, q_out_att_dual.T)
        q_out_q_dual    = riem_exp(q_in, q_out_body_dual)
        q_out_dual      = R.from_quat(q_out_q_dual.reshape(4,))
        omega         += self.compute_ang_vel(q_in, q_out_dual, dt)

        return omega


    def timer_cb(self, event):
        # wait for first pose
        if self.current_pose is None:
            return

        # extract current position & orientation
        pos = self.current_pose.position
        x   = np.array([pos.x, pos.y, pos.z])

        ori = self.current_pose.orientation
        q   = R.from_quat([ori.x, ori.y, ori.z, ori.w])

        # linear LPV-DS
        lin_vel = self.step_linear(
            x,
            self.priors,
            self.mu_pos,
            self.sigmas,
            self.A_pos,
            self.att_pos
        )

        # angular LPV-DS
        ang_vel = self.step_angular(
            q,
            self.K,
            self.priors,
            self.mu_ori,
            self.sigmas,
            self.att_ori,
            self.A_ori,
            self.dt
        )

        # publish as Twist
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.linear.z = lin_vel
        cmd.angular.x, cmd.angular.y, cmd.angular.z = ang_vel
        self.cmd_pub.publish(cmd)


if __name__ == '__main__':
    node = LPVDSNode()
    rospy.spin()
