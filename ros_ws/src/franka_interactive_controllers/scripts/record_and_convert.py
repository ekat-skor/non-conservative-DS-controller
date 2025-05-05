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
    merged_csv = f"/workspace/ros_ws/conservative_panda_ee_full_{timestamp}.csv"

    topic_csvs = {
        "/franka_state_controller/F_ext": f"...",
        # CHANGE THIS LINE IF RUNNING C VS NC
        "/nc_passive_ds_impedance_controller/ee_velocity": f"...",
        "/franka_state_controller/ee_pose": f"/workspace/ros_ws/ee_pose_{timestamp}.csv",
        "/passiveDS/desired_twist": f"..."
    }


    # Step 1: Start recording the bag file
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

    # Step 2: Extract all topics to individual CSVs
    for topic, filename in topic_csvs.items():
        rospy.loginfo(f"Extracting {topic} to {filename}...")
        with open(filename, "w") as f_out:
            subprocess.run(["rostopic", "echo", "-b", bag_name, "-p", topic], stdout=f_out)


   # Step 3: Load and merge all CSVs
    try:
        rospy.loginfo("Loading and merging CSV files...")

        # Read each topic CSV and label the columns
        def load_and_label(path, prefix):
            df = pd.read_csv(path).rename(columns={"%time": "time"})
            df['time'] = pd.to_numeric(df['time'], errors='coerce')
            df.dropna(subset=['time'], inplace=True)
            df = df.sort_values("time")
            df = df.add_prefix(prefix)
            df = df.rename(columns={f"{prefix}time": "time"})  # Keep 'time' as shared key
            return df

        df_fext  = load_and_label(topic_csvs["/franka_state_controller/F_ext"], "fext_")
        # CHANGE THIS LINE IF RUNNING C VS NC
        df_vel   = load_and_label(topic_csvs["/nc_passive_ds_impedance_controller/ee_velocity"], "ee_vel_")
        df_pose  = load_and_label(topic_csvs["/franka_state_controller/ee_pose"], "ee_pose_")
        df_twist = load_and_label(topic_csvs["/passiveDS/desired_twist"], "twist_cmd_")

        # Merge all on time using nearest match
        df_merged = pd.merge_asof(df_fext, df_vel, on="time", direction="nearest")
        df_merged = pd.merge_asof(df_merged, df_pose, on="time", direction="nearest")
        df_merged = pd.merge_asof(df_merged, df_twist, on="time", direction="nearest")

        # Save the merged CSV
        df_merged.to_csv(merged_csv, index=False)
        rospy.loginfo(f"✅ Merged CSV saved to: {merged_csv}")

        # Cleanup temporary CSVs
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
