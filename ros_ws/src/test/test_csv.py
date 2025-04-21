#!/usr/bin/env python
import rospy
import csv
import os
from geometry_msgs.msg import Twist

def read_csv_data(csv_file):
    """
    Reads CSV data and returns a list of rows, each row is a dictionary with keys:
    'time', 'v_x', 'v_y', 'v_z', 'w_x', 'w_y', 'w_z'.
    Any extraneous bracket characters are removed before conversion.
    """
    data = []
    with open(csv_file, mode='r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            new_row = {}
            for key, value in row.items():
                # Remove surrounding brackets if present and then convert
                val_str = value.strip()
                if val_str.startswith('[') and val_str.endswith(']'):
                    val_str = val_str[1:-1]  # remove the first and last char
                new_row[key] = float(val_str)
            data.append(new_row)
    return data

def publish_twist(pub, v, w):
    """Publish a Twist message given linear (v) and angular (w) velocities."""
    twist = Twist()
    twist.linear.x = v[0]
    twist.linear.y = v[1]
    twist.linear.z = v[2]
    twist.angular.x = w[0]
    twist.angular.y = w[1]
    twist.angular.z = w[2]
    pub.publish(twist)

def main():
    rospy.init_node('csv_twist_publisher', anonymous=True)
    pub = rospy.Publisher('/passiveDS/desired_twist', Twist, queue_size=10)

    # Specify CSV file path (adjust if necessary)
    csv_file = "u_shape.csv"
    if not os.path.exists(csv_file):
        rospy.logerr("CSV file '%s' does not exist. Run main.py first to generate the file." % csv_file)
        return

    # Read CSV data using the updated reader
    data = read_csv_data(csv_file)

    # Determine publishing rate from the time stamps
    if len(data) < 2:
        rospy.logerr("Not enough data in CSV file to determine publishing rate.")
        return

    dt = data[1]['time'] - data[0]['time']
    rate = rospy.Rate(1.0 / dt)

    rospy.loginfo("Starting CSV Twist publisher with dt=%.4f", dt)

    for row in data:
        if rospy.is_shutdown():
            break

        # Extract linear and angular velocities from the row
        v = (row['v_x'], row['v_y'], row['v_z'])
        w = (row['w_x'], row['w_y'], row['w_z'])
        publish_twist(pub, v, w)
        rospy.loginfo("Published Twist: linear=%s, angular=%s", v, w)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
