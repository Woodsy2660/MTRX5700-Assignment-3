#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

"""
Subscribes to an odometry topic and republishes the twist componenet of the msg on the control topic.
This is used when the simulation is active to reflect an actual control message that would be sent to the robot. No noise
is added to the message
"""

class OdomToTwistRepublisher(Node):

    def __init__(self) -> None:
        super().__init__("odom_to_control")
        self.control_pub = self.create_publisher(Twist,"~/control", 1)
        self.frame = self.declare_parameter("robot_frame", "base_link").value
        self.seq = 0
        self.create_subscription(Odometry, "~/odom", self.odomCallback, 1)

    def odomCallback(self, data: Odometry):
        self.control_pub.publish(data.twist.twist)

if __name__ == '__main__':
    rclpy.init()
    node = OdomToTwistRepublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
    