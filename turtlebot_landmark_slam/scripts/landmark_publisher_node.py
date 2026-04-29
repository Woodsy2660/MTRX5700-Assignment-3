#!/usr/bin/env python3
# ROS2 landmark publisher node - runs on the Linux TurtleBot laptop
#
# Subscribes to:
#   /camera/image_raw   - camera frames
#   /pointcloud2d       - lidar scans
#
# Publishes to:
#   /landmarks          - LandmarksMsg containing one LandmarkMsg per detected tag
#                         fields: label (tag ID), x, y (robot body frame), s_x, s_y
#
# Usage: ros2 run <your_package> landmark_publisher_node

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud
from cv_bridge import CvBridge

from landmarks_msg.msg import LandmarkMsg, LandmarksMsg


class LandmarkPublisherNode(Node):

    # from Assignment 2 calibration.py (K matrix)
    FX = 483.04   # K[0][0]
    CX = 307.23   # K[0][2]

    def __init__(self):
        super().__init__("landmark_publisher")

        from turtlebot_landmark_slam.landmark_detector import LandmarkDetector
        self.detector = LandmarkDetector(fx=self.FX, cx=self.CX)
        self.bridge   = CvBridge()

        # store the latest lidar scan so the camera callback can use it
        self._latest_pointcloud: np.ndarray | None = None

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image,      "/camera/image_raw", self._on_image,      qos)
        self.create_subscription(PointCloud, "/pointcloud2d",     self._on_pointcloud, qos)

        self._pub = self.create_publisher(LandmarksMsg, "/landmarks", 10)
        self.get_logger().info("landmark_publisher started")

    def _on_pointcloud(self, msg: PointCloud) -> None:
        if not msg.points:
            return
        self._latest_pointcloud = np.array(
            [[p.x, p.y] for p in msg.points], dtype=np.float32
        )

    def _on_image(self, msg: Image) -> None:
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge: {e}")
            return

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        observations = self.detector.process_image(
            img_bgr, timestamp_s=t, pointcloud_xy=self._latest_pointcloud
        )

        # only publish detections where we got a valid lidar range
        valid = [obs for obs in observations if obs.range_m > 0]
        if not valid:
            return

        out_msg = LandmarksMsg()
        out_msg.landmarks = []

        for obs in valid:
            lm        = LandmarkMsg()
            lm.label  = obs.tag_id
            lm.x      = obs.x
            lm.y      = obs.y
            lm.s_x    = obs.s_x
            lm.s_y    = obs.s_y
            out_msg.landmarks.append(lm)

            self.get_logger().info(
                f"id={obs.tag_id}  x={obs.x:.2f}  y={obs.y:.2f}  "
                f"s_x={obs.s_x:.3f}  s_y={obs.s_y:.3f}  "
                f"{'NEW' if obs.is_new else 'revisit'}"
            )

        self._pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LandmarkPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
