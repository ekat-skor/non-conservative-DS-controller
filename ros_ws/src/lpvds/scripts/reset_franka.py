#!/usr/bin/env python
"""
Simple script to reset the Panda in Gazebo to its initial joint configuration
and ensure franka_state_controller is loaded so /franka_state_controller/ee_pose publishes.
"""
import rospy
from controller_manager_msgs.srv import LoadController, SwitchController
from gazebo_msgs.srv import SetModelConfiguration

if __name__ == '__main__':
    rospy.init_node('reset_franka_state_and_model')

    # 1) Load & start the franka_state_controller
    rospy.loginfo("Waiting for controller_manager services...")
    rospy.wait_for_service('/controller_manager/load_controller')
    rospy.wait_for_service('/controller_manager/switch_controller')
    try:
        load_ctrl = rospy.ServiceProxy('/controller_manager/load_controller', LoadController)
        switch_ctrl = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)

        rospy.loginfo("Loading franka_state_controller...")
        res_load = load_ctrl(name='franka_state_controller')
        if not res_load.ok:
            rospy.logwarn("franka_state_controller load response not ok")

        rospy.loginfo("Starting franka_state_controller...")
        res_switch = switch_ctrl(start_controllers=['franka_state_controller'],
                                  stop_controllers=[],
                                  strictness=2)
        if not res_switch.ok:
            rospy.logwarn("franka_state_controller switch response not ok")

        rospy.loginfo("franka_state_controller is loaded and started.")
    except rospy.ServiceException as e:
        rospy.logerr(f"Controller service call failed: {e}")
        exit(1)

    # 2) Reset Panda joints via Gazebo service
    rospy.loginfo("Waiting for Gazebo set_model_configuration service...")
    rospy.wait_for_service('/gazebo/set_model_configuration')
    try:
        set_model_cfg = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
        rospy.loginfo("Resetting Panda joint configuration in Gazebo...")
        res = set_model_cfg(
            model_name='panda',
            urdf_param_name='robot_description',
            joint_names=[
                'panda_joint1','panda_joint2','panda_joint3',
                'panda_joint4','panda_joint5','panda_joint6','panda_joint7',
                'panda_finger_joint1','panda_finger_joint2'
            ],
            joint_positions=[
                0.21392,  0.35085, -0.39038, -2.02796,  0.18411,  2.34579,  2.07179
            ]
        )
        if not res.success:
            rospy.logwarn(f"SetModelConfiguration failed: {res.status_message}")
        else:
            rospy.loginfo("Panda joint configuration reset successfully.")
    except rospy.ServiceException as e:
        rospy.logerr(f"Gazebo service call failed: {e}")
        exit(1)

    rospy.loginfo("Reset sequence complete.")
