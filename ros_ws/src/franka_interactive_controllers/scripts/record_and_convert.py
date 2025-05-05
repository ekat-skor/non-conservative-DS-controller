#!/usr/bin/env python3

import rospy
import subprocess
from datetime import datetime
import pandas as pd
import os
import signal
import numpy as np
from threading import Event
import json
from geometry_msgs.msg import Pose



def main():
    rospy.init_node('record_and_convert')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_name = f"/workspace/ros_ws/panda_ee_data_{timestamp}.bag"
    merged_csv = f"/workspace/ros_ws/conservative_panda_ee_full_{timestamp}.csv"

    # Topics to record
    topic_csvs = {
        "/franka_state_controller/F_ext": f"/workspace/ros_ws/fext_{timestamp}.csv",
        "/passive_ds_impedance_controller/ee_velocity": f"/workspace/ros_ws/ee_vel_{timestamp}.csv",
        "/franka_state_controller/ee_pose": f"/workspace/ros_ws/ee_pose_{timestamp}.csv",
        "/passiveDS/desired_twist": f"/workspace/ros_ws/twist_cmd_{timestamp}.csv"
    }

    record_cmd = ["rosbag", "record", "-O", bag_name] + list(topic_csvs.keys())

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

    # Load desired position from JSON
    with open('/workspace/ros_ws/src/lpvds_damm/trained_ds/j_shape_position.json', 'r') as f:
        ds_model = json.load(f)
    desired_pos = np.array(ds_model['attractor'])

    # Global variable for live actual EE position
    global actual_pos
    actual_pos = None


    def pose_callback(msg):
        # print("[DEBUG] pose_callback triggered")
        global actual_pos
        actual_pos = np.array([
            msg.position.x,
            msg.position.y,
            msg.position.z
        ])
        # print(f"[POSE CALLBACK] Actual position updated: {actual_pos}")

    rospy.Subscriber("/franka_state_controller/ee_pose", Pose, pose_callback)

    rospy.loginfo(f"Recording to {bag_name}")
    rosbag_proc = subprocess.Popen(record_cmd)

    # Wait for position convergence or Ctrl+C
    POSITION_THRESHOLD = 1e-3  # meters
    stop_event = Event()
    
    print(f"Current EE position: {actual_pos}")
    try:
        rospy.loginfo("Recording... waiting for threshold condition.")
        while not rospy.is_shutdown():
            # print(f"Current EE position: {actual_pos}")
            if actual_pos is not None:
                dist = np.linalg.norm(actual_pos - desired_pos)
                if dist < POSITION_THRESHOLD:
                    print("✅✅✅ THRESHOLD MET: ROBOT REACHED GOAL POSITION. STOPPING RECORDING. ✅✅✅")
                    stop_event.set()
                    break
            rospy.sleep(0.1)
    except rospy.ROSInterruptException:
        rospy.loginfo("Interrupted by user.")

    # Gracefully stop rosbag
    if rosbag_proc.poll() is None:
        rosbag_proc.send_signal(signal.SIGINT)
        try:
            rosbag_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            rosbag_proc.terminate()
    rospy.loginfo("Recording complete.")

    # Extract each topic to CSV
    for topic, filename in topic_csvs.items():
        rospy.loginfo(f"Extracting {topic} to {filename}...")
        with open(filename, "w") as f_out:
            subprocess.run(["rostopic", "echo", "-b", bag_name, "-p", topic], stdout=f_out)

    # Merge all CSVs
    try:
        rospy.loginfo("Merging CSV files...")

        def load_and_label(path, prefix):
            df = pd.read_csv(path).rename(columns={"%time": "time"})
            df['time'] = pd.to_numeric(df['time'], errors='coerce')
            df.dropna(subset=['time'], inplace=True)
            df = df.sort_values("time")
            df = df.add_prefix(prefix)
            df = df.rename(columns={f"{prefix}time": "time"})
            return df

        df_fext  = load_and_label(topic_csvs["/franka_state_controller/F_ext"], "fext_")
        df_vel   = load_and_label(topic_csvs["/passive_ds_impedance_controller/ee_velocity"], "ee_vel_")
        df_pose  = load_and_label(topic_csvs["/franka_state_controller/ee_pose"], "ee_pose_")
        df_twist = load_and_label(topic_csvs["/passiveDS/desired_twist"], "twist_cmd_")

        df_merged = pd.merge_asof(df_fext, df_vel, on="time", direction="nearest")
        df_merged = pd.merge_asof(df_merged, df_pose, on="time", direction="nearest")
        df_merged = pd.merge_asof(df_merged, df_twist, on="time", direction="nearest")

        df_merged.to_csv(merged_csv, index=False)
        rospy.loginfo(f"✅ Merged CSV saved to: {merged_csv}")

        for file in topic_csvs.values():
            try:
                os.remove(file)
                rospy.loginfo(f"Deleted intermediate file: {file}")
            except FileNotFoundError:
                rospy.logwarn(f"File not found for deletion: {file}")

    except Exception as e:
        rospy.logerr(f"❌ Failed to merge CSVs: {e}")

if __name__ == "__main__":
    main()
