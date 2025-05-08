#!/usr/bin/env python3
"""
DS Twist Publisher ROS Node

This node subscribes to the end-effector pose and publishes a Twist message
computed by a linear dynamical system (DS): x_dot = A * (x - x*).
"""

import rospy
from geometry_msgs.msg import Pose, Twist
import numpy as np

class DSTwistPublisher:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('ds_twist_publisher')

        # Load attractor and DS matrix from parameters (with defaults)
        self.x_star = np.array(
            rospy.get_param(
                '~x_star',
                [5.893022010703901081e-01,-9.550721455301605656e-02,1.827824168919514880e-01] # start position for j-shape 

            )
        )
        A_param = rospy.get_param(
            '~A',
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        )
        self.A = np.array(A_param)

        # State storage
        self.current_pose = None

        # Publisher for the computed DS velocity
        self.pub_twist = rospy.Publisher(
            '/passiveDS/desired_twist', Twist, queue_size=10
        )

        # Subscriber to end-effector pose
        rospy.Subscriber(
            '/franka_state_controller/ee_pose', Pose,
            self.pose_callback, queue_size=1
        )

        # Loop rate
        self.rate = rospy.Rate(rospy.get_param('~rate_hz', 100))

        rospy.loginfo("[ds_twist_publisher] Node initialized.")

    def pose_callback(self, msg):
        # Store the current end-effector position
        self.current_pose = np.array([
            msg.position.x,
            msg.position.y,
            msg.position.z
        ])

    def spin(self):
        rospy.loginfo("[ds_twist_publisher] Spinning...")
        while not rospy.is_shutdown():
            if self.current_pose is not None:
                # Compute DS velocity
                vel = self.A.dot(self.current_pose - self.x_star)

                # Build and publish Twist message
                twist = Twist()
                twist.linear.x = float(vel[0])
                twist.linear.y = float(vel[1])
                twist.linear.z = float(vel[2])
                twist.angular.x = 0.0
                twist.angular.y = 0.0
                twist.angular.z = 0.0

                self.pub_twist.publish(twist)

            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = DSTwistPublisher()
        node.spin()
    except rospy.ROSInterruptException:
        pass