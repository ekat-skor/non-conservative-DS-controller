#!/usr/bin/env python3

import rospy
import subprocess
from datetime import datetime
import pandas as pd
import os
import signal
import numpy as np
from threading import Event
from geometry_msgs.msg import Pose
import json



def main():
    rospy.init_node('record_nc_params')

    bag_name = f"/home/tianyu/ncds_ws/non-conservative-DS-controller/ros_ws/panda_nc_params_data_.bag"
    merged_csv = f"/home/tianyu/ncds_ws/non-conservative-DS-controller/ros_ws/nc_params_data/nc_params_full_.csv"

    # Topics to record
    
    topic_csvs = {
        "/nc_passive_ds_impedance_controller/passive_ds/alpha_": f"/home/tianyu/ncds_ws/non-conservative-DS-controller/ros_ws/alpha_.csv",
        "/nc_passive_ds_impedance_controller/passive_ds/beta_r_": f"/home/tianyu/ncds_ws/non-conservative-DS-controller/ros_ws/betar_.csv",
        "/nc_passive_ds_impedance_controller/passive_ds/beta_s_": f"/home/tianyu/ncds_ws/non-conservative-DS-controller/ros_ws/betas_.csv",
        "/nc_passive_ds_impedance_controller/passive_ds/s_": f"/home/tianyu/ncds_ws/non-conservative-DS-controller/ros_ws/s_.csv",
        "/nc_passive_ds_impedance_controller/passive_ds/sdot_": f"/home/tianyu/ncds_ws/non-conservative-DS-controller/ros_ws/sdot_.csv",
        "/nc_passive_ds_impedance_controller/passive_ds/z_": f"/home/tianyu/ncds_ws/non-conservative-DS-controller/ros_ws/z_.csv",
    }

    record_cmd = ["rosbag", "record", "-O", bag_name] + list(topic_csvs.keys())

    # load target position
    with open('/home/tianyu/ncds_ws/non-conservative-DS-controller/ros_ws/src/lpvds_damm/trained_ds/j_shape_position.json', 'r') as f:
        desired_pos = np.array(json.load(f)["attractor"])

    actual_pos = None
    def pose_callback(msg):
        nonlocal actual_pos
        actual_pos = np.array([msg.position.x, msg.position.y, msg.position.z])

    rospy.Subscriber("/franka_state_controller/ee_pose", Pose, pose_callback)

    def wait_for_active_topic(topic, timeout=10):
        rospy.loginfo(f"🔄 Waiting for topic to become active: {topic}")
        start_time = rospy.Time.now()
        while not rospy.is_shutdown():
            if topic in [t[0] for t in rospy.get_published_topics()]:
                rospy.loginfo(f"✅ Topic {topic} is now active.")
                return True
            if (rospy.Time.now() - start_time).to_sec() > timeout:
                rospy.logwarn(f"⚠️ Timeout waiting for {topic}. Starting anyway.")
                return False
            rospy.sleep(0.1)

    wait_for_active_topic("/nc_passive_ds_impedance_controller/ee_velocity", timeout=10)

    rospy.loginfo(f"🎥 Starting recording to {bag_name}")
    rosbag_proc = subprocess.Popen(record_cmd)

    rospy.loginfo("Waiting 3 seconds before checking position threshold...")
    rospy.sleep(3)

    POSITION_THRESHOLD = 1e-3
    try:
        rospy.loginfo("Recording... waiting for threshold condition.")
        while not rospy.is_shutdown():
            if actual_pos is not None and np.linalg.norm(actual_pos - desired_pos) < POSITION_THRESHOLD:
                rospy.loginfo("✅ Threshold reached; stopping.")
                break
            rospy.sleep(0.1)
    except rospy.ROSInterruptException:
        pass

    # stop rosbag
    if rosbag_proc.poll() is None:
        rosbag_proc.send_signal(signal.SIGINT)
        try:
            rosbag_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            rosbag_proc.terminate()

    rospy.loginfo("Recording complete.")

    # extract topics
    for topic, filename in topic_csvs.items():
        rospy.loginfo(f"Extracting {topic} → {filename}")
        with open(filename, "w") as f_out:
            subprocess.run(["rostopic", "echo", "-b", bag_name, "-p", topic], stdout=f_out)

    # small pause to flush files
    rospy.sleep(2)

    def load_and_label(path, prefix):
        if os.stat(path).st_size == 0:
            rospy.logwarn(f"⚠️ Skipping empty file: {path}")
            return None
        df = pd.read_csv(path).rename(columns={"%time": "time"})
        df['time'] = pd.to_numeric(df['time'], errors='coerce').astype(np.float64)
        df.dropna(subset=['time'], inplace=True)
        df.sort_values("time", inplace=True)
        df = df.add_prefix(prefix)
        df.rename(columns={f"{prefix}time": "time"}, inplace=True)
        return df

    try:
        rospy.loginfo("Merging CSV files...")
        df_list = []
        for path in topic_csvs.values():
            # derive prefix from filename, keep full stem before final underscore
            stem = os.path.splitext(os.path.basename(path))[0]
            prefix = stem.rsplit("_", 1)[0] + "_"
            df = load_and_label(path, prefix)
            if df is not None:
                df_list.append(df)

        if df_list:
            df_merged = df_list[0]
            for df in df_list[1:]:
                df_merged = pd.merge_asof(df_merged, df, on="time", direction="nearest")
            df_merged.to_csv(merged_csv, index=False)
            rospy.loginfo(f"✅ Merged CSV saved to: {merged_csv}")
        else:
            rospy.logwarn("⚠️ No valid CSV files to merge.")
    except Exception as e:
        rospy.logerr(f"❌ Failed to merge CSVs: {e}")

if __name__ == "__main__":
    main()
