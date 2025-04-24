#!/usr/bin/env python
"""
test_linear.py

This ROS node implements a 3D dynamical system (DS) with a global attractor and
publishes the computed linear velocities as a Twist message to the low-level 
passive DS impedance controller (which listens to "/passiveDS/desired_twist").

The DS is given by:
    x_dot = A_DS * (x - x*)
where A_DS can be a single matrix or, for instance, a weighted combination:
    A_DS = gamma1*A1 + gamma2*A2

The global attractor (x*) is set to the origin by default.


"""

import rospy
import numpy as np
from geometry_msgs.msg import Twist

def main():
    rospy.init_node("ds_twist_publisher", anonymous=True)
    
    # Publisher to send the computed twist to the low-level controller.
    pub_twist = rospy.Publisher("/passiveDS/desired_twist", Twist, queue_size=10)
    
    # -------------------------- DS Parameters --------------------------
    ds_case = 1  # Choose ds_case: 1 for a simple linear DS, 2 for a weighted combination.
    
    # Global attractor (x*): by default at the origin.
    # x_star = np.array([0.5, 0.3, 0.5])  
    x_star = np.array([0.56245899, -0.03973889,  0.17202399])
    
    if ds_case == 1:
        # Define a linear DS: x_dot = A*(x-x*)
        A = np.array([[-1,  0,  0],
                      [ 0, -1,  0],
                      [ 0,  0, -1]])
        ds_fun = lambda x: A @ (x - x_star)
    elif ds_case == 2:
        # Weighted DS: x_dot = (gamma1*A1 + gamma2*A2)*(x-x*)
        gamma1 = 0.5
        gamma2 = 0.5
        # Example matrices A1 and A2; modify them as desired.
        A1 = np.array([[-1, 0, 0],
                       [0, -1, 0],
                       [0, 0, -1]])
        A2 = np.array([[-1, 0, 0],
                       [0, -1, 0],
                       [0, 0, -1]])
        A_DS = gamma1 * A1 + gamma2 * A2
        ds_fun = lambda x: A_DS @ (x - x_star)
    else:
        rospy.logerr("Invalid ds_case selected")
        return

    # ----------------------- Integration Setup -------------------------
    rate = rospy.Rate(100)   # Loop at 100 Hz.
    dt = 1.0 / 100.0         # Time step for integration.
    
    # Initial DS state (virtual command position) - change to your start position.
    current_state = np.array([0.5, 0.5, 0.5])
    
    rospy.loginfo("Starting DS Twist Publisher...")
    
    while not rospy.is_shutdown():
        # Compute DS velocity: v = A_DS*(x - x*)
        vel = ds_fun(current_state)
        
        # Update the virtual state for the next iteration.
        current_state = current_state + vel * dt
        
        # Build the Twist message.
        twist_msg = Twist()
        twist_msg.linear.x = vel[0]
        twist_msg.linear.y = vel[1]
        twist_msg.linear.z = vel[2]
        # Angular velocities remain zero in this example.
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
        twist_msg.angular.z = 0.0
        
        # Publish the Twist message for the passive DS impedance controller.
        pub_twist.publish(twist_msg)
        
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
