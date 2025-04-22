#!/usr/bin/env python
import rospy
import numpy as np
import matplotlib
# Set the backend (if needed for visualization elsewhere)
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Twist

# Import your helper modules (make sure the paths are correct)
from src.se3_class import se3_class
from src.util import plot_tools, load_tools, process_tools

def main():
    # Initialize the ROS node
    rospy.init_node('lpvds_publisher', anonymous=True)
    
    # Create a publisher to send Twist commands to the low-level controller.
    # You can modify the topic name as needed.
    twist_pub = rospy.Publisher('/low_level_controller/twist', Twist, queue_size=10)
    
    # Set the rate at which to publish twist messages (e.g., 100 Hz)
    rate = rospy.Rate(100) 
    
    '''Load data'''
    # You can choose your dataset here; for example, loading the demo dataset.
    p_raw, q_raw, t_raw, dt = load_tools.load_demo_dataset()
    
    '''Process data'''
    p_in, q_in, t_in = process_tools.pre_process(p_raw, q_raw, t_raw, opt="savgol")
    p_out, q_out = process_tools.compute_output(p_in, q_in, t_in)
    p_init, q_init, p_att, q_att = process_tools.extract_state(p_in, q_in)
    p_in, q_in, p_out, q_out = process_tools.rollout_list(p_in, q_in, p_out, q_out)
    
    '''Run lpvds'''
    # Initialize and begin the LPV-DS. The se3_class contains your dynamics.
    se3_obj = se3_class(p_in, q_in, p_out, q_out, p_att, q_att, dt, K_init=4)
    se3_obj.begin()
    
    '''Evaluate results via simulation'''
    # Use the first trajectory for initial conditions.
    p_initial = p_init[0]
    # Convert the initial orientation from the provided structure into a scipy Rotation.
    q_initial = R.from_quat(q_init[0].as_quat())
    
    # Run the simulation with the LPV-DS and return trajectories and twist information.
    # Here, 'v_test' is an array of linear velocities and 'w_test' is an array of angular velocities.
    p_test, q_test, gamma_test, v_test, w_test = se3_obj.sim(p_initial, q_initial, step_size=0.01)
    
    rospy.loginfo("Starting to publish twist commands to low-level controller...")
    
    # Publish twist commands for each time step simulated.
    num_steps = len(v_test)
    for i in range(num_steps):
        if rospy.is_shutdown():
            break
        
        # Create a new Twist message for the current step
        twist_msg = Twist()
        # Linear velocity: set based on v_test (x, y, z)
        twist_msg.linear.x = v_test[i][0]
        twist_msg.linear.y = v_test[i][1]
        twist_msg.linear.z = v_test[i][2]
        # Angular velocity: set based on w_test (roll, pitch, yaw rates)
        twist_msg.angular.x = w_test[i][0]
        twist_msg.angular.y = w_test[i][1]
        twist_msg.angular.z = w_test[i][2]
        
        # Publish the twist message
        twist_pub.publish(twist_msg)
        rate.sleep()
    
    rospy.loginfo("Twist command publishing complete.")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
