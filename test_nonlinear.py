#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import tf.transformations as tf_trans  

class LPVDSNonlinear3D(object):
    """
    ROS node that generates a 3D nonlinear DS producing both linear and angular commands.
    It publishes:
      - 'desired_twist' as geometry_msgs/Twist (linear and angular velocities), and
      - 'desired_damp_eigval' as std_msgs/Float32 (for linear DS damping).
      
    The node maintains the orientation state as a quaternion and computes the angular velocity
    from the quaternion error between the desired and current orientation.
    
    It also records and plots the 3D position trajectory when shutting down.
    """

    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('lpv_ds_nonlin_3d', anonymous=True)
        
        # Register shutdown hook to plot the trajectory
        rospy.on_shutdown(self.plot_trajectory)
        
        # Create publishers for linear/angular commands and damping value
        self.pub_desired_twist = rospy.Publisher(
            "/passiveDS/desired_twist",
            Twist,
            queue_size=20
        )
        self.pub_desired_damp = rospy.Publisher(
            "/passiveDS/desired_damp_eigval",
            Float32,
            queue_size=20
        )

        # Set publish rate (Hz)
        self.publish_rate = 100.0
        self.rate = rospy.Rate(self.publish_rate)

        # Define desired goal position (x, y, z)
        self.x_goal = np.array([0.4, -0.3, 0.2])
        # Initial position (for position DS)
        self.x = np.array([-0.5, 0.0, -0.5])
        self.dt = 1.0 / self.publish_rate

        # Controller gains for linear DS
        self.alpha = 2.0  # Linear attractor gain
        self.beta  = 3.0  # Swirl/spiral term gain
        self.gamma = 0.5  # Damping modulation factor

        # Orientation state as quaternion [x, y, z, w]
        # Start with the identity quaternion (no rotation)
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])
        # Define desired orientation as a quaternion
        # Example: 90° yaw rotation. Use tf_trans.quaternion_from_euler(roll, pitch, yaw)
        self.orientation_goal = tf_trans.quaternion_from_euler(0.0, 0.0, np.pi/2)
        # Gain for the orientation DS (proportional controller)
        self.k_orientation = 2.0

        # List for recording the position trajectory for visualization
        self.trajectory = []

        self.run()

    def run(self):
        """
        Main loop that publishes twist (linear and angular) and damping messages,
        while updating both position and orientation.
        """
        while not rospy.is_shutdown():
            # Compute linear velocity using the 3D position DS
            velocity = self.compute_velocity(self.x)
            # Compute angular velocity from quaternion error (in rad/s)
            ang_velocity = self.compute_angular_velocity(self.orientation)

            # Create and publish the Twist message (both linear and angular parts)
            twist_msg = Twist()
            # Linear velocities
            twist_msg.linear.x = velocity[0]
            twist_msg.linear.y = velocity[1]
            twist_msg.linear.z = velocity[2]
            # Angular velocities
            twist_msg.angular.x = ang_velocity[0]
            twist_msg.angular.y = ang_velocity[1]
            twist_msg.angular.z = ang_velocity[2]
            self.pub_desired_twist.publish(twist_msg)

            # Compute a scalar damping value for the linear DS and publish it
            damp_msg = Float32()
            damp_value = self.compute_damping(self.x)
            damp_msg.data = damp_value
            self.pub_desired_damp.publish(damp_msg)

            # Record the position for trajectory plotting
            self.trajectory.append(self.x.copy())

            # Update position using Euler integration
            self.x += velocity * self.dt
            # Update orientation based on computed angular velocity
            self.orientation = self.integrate_quaternion(self.orientation, ang_velocity, self.dt)

            self.rate.sleep()

    def compute_velocity(self, x):
        """
        Compute the 3D linear velocity command.
        Combines a linear attractor with a swirl term.
        """
        x_to_goal = self.x_goal - x
        dist = np.linalg.norm(x_to_goal)
        rho = np.exp(-5.0 * dist**2)  # gating function to reduce swirl near the goal
        linear_part = self.alpha * x_to_goal
        swirl_axis = np.array([0, 0, 1])
        swirl_dir = np.cross(swirl_axis, x_to_goal)
        swirl_part = self.beta * rho * swirl_dir

        velocity = linear_part + swirl_part
        max_vel = 1.0
        speed = np.linalg.norm(velocity)
        if speed > max_vel:
            velocity = (velocity / speed) * max_vel

        return velocity

    def compute_angular_velocity(self, q_current):
        """
        Compute the angular velocity command based on the quaternion error.
        The error quaternion is computed as:
            q_error = q_desired * inverse(q_current)
        and then converted to axis-angle form. The angular velocity is then:
            ang_velocity = k_orientation * theta * axis
        where theta is the rotation angle and axis is the normalized rotation axis.
        """
        # Compute the error quaternion: q_error = q_goal * conjugate(q_current)
        q_error = tf_trans.quaternion_multiply(self.orientation_goal, tf_trans.quaternion_conjugate(q_current))

        # Ensure q_error is normalized
        q_error = q_error / np.linalg.norm(q_error)

        # Compute the rotation angle (theta)
        theta = 2.0 * np.arccos(np.clip(q_error[3], -1.0, 1.0))
        
        # To avoid numerical issues, check the sine of half the angle
        sin_half_theta = np.sqrt(1.0 - q_error[3] * q_error[3])
        if sin_half_theta < 1e-6:
            axis = np.array([0.0, 0.0, 0.0])
        else:
            axis = q_error[0:3] / sin_half_theta

        # Proportional control: angular velocity is gain times the angle error along the axis.
        ang_velocity = self.k_orientation * theta * axis

        # Optionally, saturate the angular velocity
        max_ang_vel = 1.0
        norm_ang = np.linalg.norm(ang_velocity)
        if norm_ang > max_ang_vel:
            ang_velocity = (ang_velocity / norm_ang) * max_ang_vel

        return ang_velocity

    def integrate_quaternion(self, q, ang_velocity, dt):
        """
        Update the quaternion given the current quaternion q, the angular velocity vector (rad/s),
        and the timestep dt. This is done by constructing a small rotation quaternion delta_q and
        multiplying it with the current orientation.
        """
        # Compute the rotation magnitude over the timestep
        omega_norm = np.linalg.norm(ang_velocity)
        if omega_norm < 1e-6:
            return q  # no significant rotation
        
        # Rotation angle for the timestep
        theta = omega_norm * dt
        # Normalize the angular velocity vector to obtain the rotation axis
        axis = ang_velocity / omega_norm
        
        # Construct the quaternion for the rotation delta_q:
        # delta_q = [axis*sin(theta/2), cos(theta/2)]
        delta_q = np.zeros(4)
        delta_q[0:3] = axis * np.sin(theta / 2.0)
        delta_q[3] = np.cos(theta / 2.0)

        # Update the current quaternion using quaternion multiplication:
        q_new = tf_trans.quaternion_multiply(delta_q, q)
        # Normalize the resulting quaternion
        q_new = q_new / np.linalg.norm(q_new)
        return q_new

    def compute_damping(self, x):
        """
        Compute a scalar damping value that increases with distance from the goal.
        """
        dist_to_goal = np.linalg.norm(self.x_goal - x)
        base_damp = 2.0   # Minimum damping
        dist_factor = 5.0 # Scale factor with distance
        damping_value = base_damp + self.gamma * dist_factor * dist_to_goal
        return damping_value

    def plot_trajectory(self):
        """
        Plot the recorded 3D trajectory when the node shuts down.
        """
        if not self.trajectory:
            rospy.loginfo("No trajectory data to plot.")
            return

        rospy.loginfo("Plotting trajectory...")

        traj = np.array(self.trajectory)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label='Nonlinear DS Trajectory', linewidth=2)
        ax.scatter(self.x_goal[0], self.x_goal[1], self.x_goal[2], c='r', marker='*', s=200, label='Goal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Trajectory of the LPV-DS Nonlinear Controller')
        ax.legend()
        plt.show()

if __name__ == '__main__':
    try:
        LPVDSNonlinear3D()
    except rospy.ROSInterruptException:
        pass




















