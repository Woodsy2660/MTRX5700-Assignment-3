from abc import ABC, abstractmethod
from typing import Callable, Tuple
import numpy as np
from rclpy.node import Node

from turtlebot_landmark_slam.types import ControlMeasurement, LandmarkMeasurement

from landmarks_msg.msg import LandmarksMsg
from geometry_msgs.msg import Twist

#function which takes ControlMeasurement as an argument and has void (none) return type
ControlHandler = Callable[[ControlMeasurement], None]
#function which takes LandmarkMeasurement as an argument and has void (none) return type
LandmarkHandler = Callable[[LandmarkMeasurement], None]

class DataProviderBase(ABC):

    def __init__(
        self,
        node: Node,
        control_handler: ControlHandler,
        landmark_handler: LandmarkHandler,
        **kwargs,
    ) -> None:
        self._node = node
        self._control_handler = control_handler
        self._landmark_handler = landmark_handler

        self._last_control_msg_time = None
        self._last_landmark_msg_time = None
        self._last_small_motion_log_time = None

        self.std_dev_linear_vel = float(
            self._node.declare_parameter("std_dev_linear_vel", 0.01).value
        )
        self.std_dev_angular_vel = float(
            self._node.declare_parameter("std_dev_angular_vel", (5 * np.pi) / 180).value
        )

        self._node.get_logger().info(
            f"[DataProvider] std_dev_linear_vel: {self.std_dev_linear_vel}"
        )
        self._node.get_logger().info(
            f"[DataProvider] std_dev_angular_vel: {self.std_dev_angular_vel}"
        )

        # Subscribe to the cylinder node output
        self._landmarks_subscription = self._node.create_subscription(
            LandmarksMsg, "/landmarks", self.landmarkCallback, 1
        )

        # Subscribe to control message. This should be a Twist
        self._control_subscription = self._node.create_subscription(
            Twist, "/control", self.controlCallback, 1
        )

    def controlCallback(self, twist: Twist):
        now = self._node.get_clock().now()

        if self._last_control_msg_time is None:
            self._last_control_msg_time = now
            return

        dt = (now - self._last_control_msg_time).nanoseconds / 1e9
        self._last_control_msg_time = now
        # read linear and angular velocities from the ROS message
        linear_vel = twist.linear.x# velocity of the robot in its instantaneous frame. X is the forward direction
        angular_vel = twist.angular.z   # angular velocity about the Z axis of the robot's instantaneous frame

         # when there is no motion do not perform a prediction
        if abs(linear_vel) < 0.009 and abs(angular_vel) < 0.09:
            should_log = (
                self._last_small_motion_log_time is None
                or (now - self._last_small_motion_log_time).nanoseconds / 1e9 >= 10.0
            )
            if should_log:
                self._node.get_logger().info(
                    "Small linear or angular motion. Skipping predict step"
                )
                self._last_small_motion_log_time = now
            return
        
        motion_command, motion_covariance = self._constructMotionWithCovariance(
            linear_vel, angular_vel, self.std_dev_linear_vel, self.std_dev_angular_vel, dt
        )

        assert(motion_command.shape == (3,1))
        assert(motion_covariance.shape == (3,3))

        dx = motion_command[0][0]
        dy = motion_command[1][0]
        dtheta = motion_command[2][0]

        self._control_handler(ControlMeasurement(dx, dy, dtheta, motion_covariance))

    def landmarkCallback(self, landmarks: LandmarksMsg):
        for landmark_msg in landmarks.landmarks:
            landmark_measurement = LandmarkMeasurement.from_landmark_msg(landmark_msg)
            self._landmark_handler(landmark_measurement)

    @abstractmethod
    def _constructMotionWithCovariance(
        self, 
        linear_vel: float, 
        angular_vel: float,
        std_dev_linear_vel: float,
        std_dev_angular_vel: float,
        dt: float) -> Tuple[np.array, np.array]:
        ...

class OnlineDataProvider(DataProviderBase):

    def __init__(
        self,
        node: Node,
        control_handler: Callable,
        landmark_handler: Callable,
        **kwargs,
    ) -> None:
        super().__init__(node, control_handler, landmark_handler, **kwargs)

    
    def _constructMotionWithCovariance(self, linear_vel: float, angular_vel: float, std_dev_linear_vel: float, std_dev_angular_vel: float, dt: float) -> Tuple[np.array, np.array]:
        s_linear_vel_x = self.std_dev_linear_vel * linear_vel * dt #  5 cm / seg 
        s_linear_vel_y = 0.000000001 # just a small value as there is no motion along y of the robot
        s_angular_vel = self.std_dev_angular_vel * angular_vel * dt  # 2 deg / seg

        # compute the motion command [dx, dy, dtheta]. On the real robot we dont add any perterbations
        # Note: this is an approximation but works as time steps are small          
        dx = linear_vel * dt + s_linear_vel_x
        dy = 0.0     # there is no motion along y of the robot
        dtheta = angular_vel * dt + s_angular_vel

        # Calculate motion command (u) and set it
        motion_command = np.array([[dx], [dy], [dtheta]])

        motion_covariance = np.array([[(s_linear_vel_x)**2, 0.0, 0.0],
                                           [0.0, (s_linear_vel_y)**2, 0.0],
                                           [0.0, 0.0, (s_angular_vel)**2]])

        return motion_command, motion_covariance
        

class SimulationDataProvider(DataProviderBase):

    def __init__(
        self,
        node: Node,
        control_handler: ControlHandler,
        landmark_handler: LandmarkHandler,
        **kwargs,
    ) -> None:
        super().__init__(node, control_handler, landmark_handler, **kwargs)

    def _constructMotionWithCovariance(self, linear_vel: float, angular_vel: float, std_dev_linear_vel: float, std_dev_angular_vel: float, dt: float) -> Tuple[np.array, np.array]:
        # Simulation: we have set s_linear_vel and s_angular_vel to be proportional to the distance or angle moved
        # In real robots this can be read directly from the odometry message if it is provided
        s_linear_vel_x = self.std_dev_linear_vel * linear_vel * dt #  5 cm / seg 
        s_linear_vel_y = 0.000000001 # just a small value as there is no motion along y of the robot
        s_angular_vel = self.std_dev_angular_vel * angular_vel * dt  # 2 deg / seg

        # compute the motion command [dx, dy, dtheta] 
        # Simulation: here we add the random gaussian noise proportional to the distance travelled and angle rotated.
        # Note: this is an approximation but works as time steps are small          
        dx = linear_vel * dt + s_linear_vel_x * np.random.standard_normal()
        dy = 0.0     # there is no motion along y of the robot
        dtheta = angular_vel * dt + s_angular_vel * np.random.standard_normal()

        # Calculate motion command (u) and set it
        motion_command = np.array([[dx], [dy], [dtheta]])

        motion_covariance = np.array([[(s_linear_vel_x)**2, 0.0, 0.0],
                                           [0.0, (s_linear_vel_y)**2, 0.0],
                                           [0.0, 0.0, (s_angular_vel)**2]])
        return motion_command, motion_covariance