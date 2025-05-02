#!/usr/bin/env python3

import rospy
import subprocess
from datetime import datetime
import pandas as pd
import os

def main():
    rospy.init_node('record_and_convert')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_name = f"/workspace/ros_ws/panda_ee_data_{timestamp}.bag"
    pose_csv = f"/workspace/ros_ws/panda_ee_pose_{timestamp}.csv"
    twist_csv = f"/workspace/ros_ws/panda_ee_twist_{timestamp}.csv"
    merged_csv = f"/workspace/ros_ws/panda_ee_full_{timestamp}.csv"

    # Step 1: Start recording the bag file
    record_cmd = [
        "rosbag", "record", "-O", bag_name,
        "/franka_state_controller/O_T_EE",
        "/franka_state_controller/franka_states"
    ]

    def wait_for_topic(topic, timeout=30):
        rospy.loginfo(f"Waiting for topic: {topic}")
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            topics = rospy.get_published_topics()
            if topic in [t[0] for t in topics]:
                rospy.loginfo(f"Found topic: {topic}")
                return True
            if (rospy.Time.now() - start).to_sec() > timeout:
                rospy.logwarn(f"Timeout while waiting for topic {topic}")
                return False
            rospy.sleep(0.1)

    wait_for_topic("/passiveDS/desired_twist")

    rospy.loginfo(f"Recording to {bag_name}")
    rosbag_proc = subprocess.Popen(record_cmd)

    try:
        rospy.loginfo("Recording... press Ctrl+C to stop.")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Interrupted, stopping recording...")
    finally:
        rosbag_proc.terminate()
        rosbag_proc.wait()
        rospy.loginfo("Recording complete.")

    # Step 2: Extract both topics to CSV
    rospy.loginfo("Extracting pose...")
    with open(pose_csv, "w") as f_pose:
        subprocess.run(["rostopic", "echo", "-b", bag_name, "-p", "/franka_state_controller/O_T_EE"], stdout=f_pose)

    rospy.loginfo("Extracting velocity...")
    with open(twist_csv, "w") as f_twist:
        subprocess.run(["rostopic", "echo", "-b", bag_name, "-p", "/franka_state_controller/franka_states"], stdout=f_twist)

    # Step 3: Load and merge CSVs
    rospy.loginfo("Merging CSV files...")
    try:
        df_pose = pd.read_csv(pose_csv)
        df_twist = pd.read_csv(twist_csv)

        df_pose.rename(columns={"%time": "time"}, inplace=True)
        df_twist.rename(columns={"%time": "time"}, inplace=True)

        df_pose['time'] = pd.to_numeric(df_pose['time'], errors='coerce')
        df_twist['time'] = pd.to_numeric(df_twist['time'], errors='coerce')
        df_pose.dropna(subset=['time'], inplace=True)
        df_twist.dropna(subset=['time'], inplace=True)

        df_merged = pd.merge_asof(
            df_pose.sort_values('time'),
            df_twist.sort_values('time')[
                ['time',
                'field.O_dP_EE_d0', 'field.O_dP_EE_d1', 'field.O_dP_EE_d2',
                'field.O_dP_EE_c0', 'field.O_dP_EE_c1', 'field.O_dP_EE_c2']
            ],
            on='time',
            direction='nearest'
)


        df_merged.rename(columns={
            'field.O_dP_EE_d0': 'vx_desired',
            'field.O_dP_EE_d1': 'vy_desired',
            'field.O_dP_EE_d2': 'vz_desired',
            'field.O_dP_EE_c0': 'vx_actual',
            'field.O_dP_EE_c1': 'vy_actual',
            'field.O_dP_EE_c2': 'vz_actual'
        }, inplace=True)


        df_merged.to_csv(merged_csv, index=False)
        rospy.loginfo(f"Merged CSV saved to: {merged_csv}")

        os.remove(pose_csv)
        os.remove(twist_csv)

    except Exception as e:
        rospy.logerr(f"Failed to merge CSVs: {e}")

if __name__ == "__main__":
    main()
