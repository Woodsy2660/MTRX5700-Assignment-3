#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped


class PathPublisher(Node):
    """Accumulates /ekf/odom poses into /ekf/path for RViz trajectory display."""

    def __init__(self) -> None:
        super().__init__("ekf_path_publisher")
        self.path = Path()
        self.create_subscription(Odometry, "/ekf/odom", self.odom_cb, 10)
        self.path_pub = self.create_publisher(Path, "/ekf/path", 10)

    def odom_cb(self, msg: Odometry) -> None:
        # Inherit frame_id from the odom message so the path sits in the
        # same frame as the landmarks and the robot arrow.
        self.path.header = msg.header

        ps = PoseStamped()
        ps.header = msg.header
        ps.pose = msg.pose.pose
        self.path.poses.append(ps)

        self.path_pub.publish(self.path)


def main() -> None:
    rclpy.init()
    node = PathPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()