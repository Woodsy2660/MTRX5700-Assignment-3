import math

import numpy as np
import rclpy
from rclpy.node import Node
from threading import Lock

from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray

from turtlebot_landmark_slam.ekf import ExtendedKalmanFilter
from turtlebot_landmark_slam.types import LandmarkMeasurement, ControlMeasurement
from turtlebot_landmark_slam.dataprovider import *  # lazy import


class Pipeline(object):
    """Wires together the EKF with ROS2 data providers and publishers.

    Subscribes to control and landmark measurements, runs the EKF predict/update
    cycle, and publishes the estimated odometry and landmark map.
    """

    def __init__(self, node: Node, ekf: ExtendedKalmanFilter) -> None:
        self._node = node
        self._ekf = ekf
        self._last_odom_time = None
        self._lock = Lock()
        self._seen_landmarks = set()

        self.is_real = bool(self._node.declare_parameter("is_real", False).value)

        if self.is_real:
            self._node.get_logger().info(
                "is_real=true: using OnlineDataProvider (no noise added to control input)."
                " Initialising EKF starting pose from /odom."
            )
            self._initialiseStartingPose()
            self._data_provider = OnlineDataProvider(
                self._node, self.controlHandler, self.landmarkHandler
            )
        else:
            self._node.get_logger().info(
                "is_real=false: using SimulationDataProvider (noise added to control input)."
                " Initialising EKF starting pose from /odom."
            )
            self._initialiseStartingPose()
            self._data_provider = SimulationDataProvider(
                self._node, self.controlHandler, self.landmarkHandler
            )

        self.odom_publisher = self._node.create_publisher(Odometry, "/ekf/odom", 1)
        self.map_publisher = self._node.create_publisher(MarkerArray, "/ekf/map", 5)
        self.publisher_timer = self._node.create_timer(0.3, self.publishTimerCallback)

    # ------------------------------------------------------------------
    # EKF callbacks
    # ------------------------------------------------------------------

    def controlHandler(self, control_measurement: ControlMeasurement):
        """Run the EKF predict step on each incoming control measurement."""
        with self._lock:
            self._last_odom_time = self._node.get_clock().now().to_msg()
            self._ekf.predict(control_measurement)

    def landmarkHandler(self, landmark_measurement: LandmarkMeasurement):
        """Run the EKF update step on each incoming landmark measurement."""
        with self._lock:
            is_new = landmark_measurement.label not in self._seen_landmarks
            if is_new:
                self._seen_landmarks.add(landmark_measurement.label)
            self._ekf.update(landmark_measurement, is_new)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publishTimerCallback(self):
        self.publishState()

    def publishState(self):
        """Publish the current EKF state as an Odometry message and a landmark MarkerArray."""
        if self._last_odom_time is None:
            return

        self._publishOdometry()
        self._publishLandmarkMap()

    def _publishOdometry(self):
        msg = Odometry()
        msg.header.stamp = self._last_odom_time
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"

        with self._lock:
            msg.pose.pose.position.x = float(self._ekf.x[0])
            msg.pose.pose.position.y = float(self._ekf.y[0])
            msg.pose.pose.position.z = 0.0

            quat = self._quaternion_from_yaw(self._ekf.yaw)
            msg.pose.pose.orientation.x = quat[0]
            msg.pose.pose.orientation.y = quat[1]
            msg.pose.pose.orientation.z = quat[2]
            msg.pose.pose.orientation.w = quat[3]

            # The ROS2 Odometry covariance is a 6x6 matrix (row-major, 36 elements)
            # for [x, y, z, roll, pitch, yaw]. Populate the [x, y, yaw] sub-block.
            pose_cov = self._ekf.pose_covariance
            cov = np.zeros(36, dtype=np.float64)
            cov[0] = pose_cov[0, 0]  # x-x
            cov[1] = pose_cov[0, 1]  # x-y
            cov[5] = pose_cov[0, 2]  # x-yaw
            cov[6] = pose_cov[1, 0]  # y-x
            cov[7] = pose_cov[1, 1]  # y-y
            cov[11] = pose_cov[1, 2]  # y-yaw
            cov[30] = pose_cov[2, 0]  # yaw-x
            cov[31] = pose_cov[2, 1]  # yaw-y
            cov[35] = pose_cov[2, 2]  # yaw-yaw
            msg.pose.covariance = cov

        self.odom_publisher.publish(msg)

    def _publishLandmarkMap(self):
        landmark_poses = self._ekf.state_mean[3:].flatten()

        # diagnostic — print what we're about to publish
        print(f"[publish] state[3:] = {landmark_poses.round(4).tolist()} "
              f"len={len(landmark_poses)}",
              flush=True)

        seen_landmarks = list(self._seen_landmarks)
        marker_array_msg = MarkerArray()

        for i in range(len(landmark_poses) // 2):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.id = seen_landmarks[i]
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = float(landmark_poses[2 * i])
            marker.pose.position.y = float(landmark_poses[2 * i + 1])
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.color.r = 1.0
            marker.color.a = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.frame_locked = False
            marker_array_msg.markers.append(marker)

        self.map_publisher.publish(marker_array_msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _initialiseStartingPose(self):
        """Seed the EKF state from a single ground-truth odometry message."""
        odom = self._wait_for_message("/odom", Odometry, 10.0)

        if odom is None:
            self._node.get_logger().warn(
                "Timed out waiting for ~/gt_odom. EKF starting pose may not match"
                " the simulated robot's actual starting pose."
            )
            return

        self._node.get_logger().info("Initialising EKF starting pose from ~/gt_odom.")
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        q = odom.pose.pose.orientation
        yaw = self._yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self._ekf._state_vector = np.array([[x], [y], [yaw]])

    @staticmethod
    def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _quaternion_from_yaw(yaw) -> tuple:
        """Convert a yaw angle (numpy array of shape (1,)) to a (x, y, z, w) quaternion."""
        half_yaw = yaw[0] * 0.5
        return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))

    def _wait_for_message(self, topic: str, msg_type, timeout_sec: float):
        """Spin until a single message arrives on `topic` or the timeout elapses."""
        result = {"msg": None}

        def _callback(msg):
            result["msg"] = msg

        subscription = self._node.create_subscription(msg_type, topic, _callback, 1)
        start_time = self._node.get_clock().now()
        while (
            result["msg"] is None
            and (self._node.get_clock().now() - start_time).nanoseconds / 1e9
            < timeout_sec
        ):
            rclpy.spin_once(self._node, timeout_sec=0.1)

        self._node.destroy_subscription(subscription)
        return result["msg"]
