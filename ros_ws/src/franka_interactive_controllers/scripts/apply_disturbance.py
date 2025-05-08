#!/usr/bin/env python3

import rospy
import time
from geometry_msgs.msg import Wrench, Point
from gazebo_msgs.srv import ApplyBodyWrench, ApplyBodyWrenchRequest

def main():
    try:
        rospy.loginfo("🚀 Starting apply_disturbance node...")
        rospy.init_node("disturbance_applier", anonymous=True)
        rospy.loginfo("🟢 Node initialized.")
        
        rospy.wait_for_service("/gazebo/apply_body_wrench")
        rospy.loginfo("🛠️ Wrench service is available.")

        apply_wrench = rospy.ServiceProxy("/gazebo/apply_body_wrench", ApplyBodyWrench)

        rospy.loginfo("⏳ Waiting 5 seconds before applying disturbance...")
        time.sleep(4.0)

        req = ApplyBodyWrenchRequest()
        req.body_name = "panda::panda_hand"  # Apply to end-effector
        req.reference_frame = "panda::panda_hand"  # Apply in local frame
        req.reference_point = Point(0.0, 0.0, 0.0)
        req.wrench.force.x = 40.0
        req.wrench.force.y = 10.0
        req.wrench.force.z = 0.0
        req.duration.nsecs = int(1 * 1e9)  # Apply for 0.65 seconds

        apply_wrench(req)
        rospy.loginfo("✅ EXTERNAL DISTURBANCE APPLIED TO PANDA's HAND.")
    except Exception as e:
        rospy.logerr(f"❌ ERROR IN DISTURBANCE SCRIPT: {e}")

if __name__ == "__main__":
    main()
