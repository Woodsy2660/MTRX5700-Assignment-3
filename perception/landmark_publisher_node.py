"""
landmark_publisher_node.py — ROS2 node for Assignment 3.

Runs on the Linux TurtleBot laptop (not the Mac).
Subscribes to /camera/image_raw and /pointcloud2d, detects ArUco tags,
computes range-bearing measurements, and publishes to /landmarks.

To use:
  1. Copy landmark_detector.py and this file onto the Linux laptop
     (into your ROS2 package's directory alongside your other nodes)
  2. Check the LandmarkMsg field names match your landmarks_msgs package
  3. Replace FX / CX with your Assignment 2 calibration values
  4. Run: ros2 run <your_package> landmark_publisher_node

Dependencies (already in your ROS2 workspace):
  - rclpy
  - sensor_msgs
  - landmarks_msgs  (the custom message from the ekf-landmark-slam scaffold)
  - cv_bridge
  - numpy, opencv-python
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud
from cv_bridge import CvBridge

# ── TODO: confirm field names by running:
#    ros2 interface show landmarks_msgs/msg/LandmarkMsg
from landmarks_msgs.msg import LandmarkMsg


class LandmarkPublisherNode(Node):

    # ── Camera intrinsics — replace with your Assignment 2 calibration ────────
    FX = 530.0      # focal length x, pixels
    CX = 320.0      # principal point x, pixels

    def __init__(self):
        super().__init__("landmark_publisher")

        # Import here so the file can be linted/imported without landmark_detector on PYTHONPATH
        from landmark_detector import LandmarkDetector
        self.detector = LandmarkDetector(fx=self.FX, cx=self.CX)

        self.bridge = CvBridge()

        # Latest lidar scan — updated in its own callback, read in camera callback
        self._latest_pointcloud: np.ndarray | None = None

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.create_subscription(Image,      "/camera/image_raw", self._on_image,      qos)
        self.create_subscription(PointCloud, "/pointcloud2d",     self._on_pointcloud, qos)

        self._pub = self.create_publisher(LandmarkMsg, "/landmarks", 10)

        self.get_logger().info("LandmarkPublisher ready — watching /camera/image_raw + /pointcloud2d")

    # ── Lidar callback ────────────────────────────────────────────────────────

    def _on_pointcloud(self, msg: PointCloud) -> None:
        """Cache the latest lidar scan as an Nx2 numpy array [x, y]."""
        if not msg.points:
            return
        pts = np.array([[p.x, p.y] for p in msg.points], dtype=np.float32)
        self._latest_pointcloud = pts

    # ── Camera callback ───────────────────────────────────────────────────────

    def _on_image(self, msg: Image) -> None:
        """Detect ArUco tags and publish a LandmarkMsg for each valid detection."""
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge failed: {e}")
            return

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        observations = self.detector.process_image(
            img_bgr,
            timestamp_s=t,
            pointcloud_xy=self._latest_pointcloud,
        )

        for obs in observations:
            lm_msg = LandmarkMsg()

            # ── TODO: set field names to match your landmarks_msgs definition ──
            # Run: ros2 interface show landmarks_msgs/msg/LandmarkMsg
            lm_msg.id      = obs.tag_id
            lm_msg.range   = obs.range_m      # metres; -1.0 if no lidar match
            lm_msg.bearing = obs.bearing      # radians, positive = right

            self._pub.publish(lm_msg)

            self.get_logger().info(
                f"Published id={obs.tag_id}  "
                f"range={obs.range_m:.2f}m  "
                f"bearing={np.rad2deg(obs.bearing):+.1f}deg  "
                f"{'NEW' if obs.is_new else 'revisit'}"
            )


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
