#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool
from gazebo_msgs.srv import ApplyBodyWrench, ApplyBodyWrenchRequest
from geometry_msgs.msg import Point, Wrench

def main():
    rospy.init_node("apply_disturbance")

    # 1) Advertise a latched Bool publisher
    apply_torque_pub = rospy.Publisher(
        '/passiveDS/apply_torque',
        Bool,
        queue_size=1,
        latch=True
    )
    # let connections be established
    rospy.sleep(0.2)

    # 2) Start with torque OFF
    apply_torque_pub.publish(Bool(data=False))
    rospy.loginfo("🔧 apply_torque = False")

    # wait however long before triggering your disturbance
    rospy.loginfo("⏳ Waiting 5 seconds before toggling torque ON → OFF")
    rospy.sleep(3.0)

    # 3) Turn it ON (controller will catch this once)
    apply_torque_pub.publish(Bool(data=True))
    rospy.loginfo("🔧 apply_torque = True")

    # small pause so the controller has time to see the True
    rospy.sleep(1.0)

    # 4) Immediately turn it back OFF
    apply_torque_pub.publish(Bool(data=False))
    rospy.loginfo("🔧 apply_torque = False again")

    # (If you also want to call ApplyBodyWrench, you can do it here,
    #  once the service is ready.)
    # rospy.wait_for_service('/gazebo/apply_body_wrench')
    # apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
    # req = ApplyBodyWrenchRequest(...)
    # apply_wrench(req)

    rospy.loginfo("✅ Disturbance toggle complete, node is done.")

if __name__ == "__main__":
    main()