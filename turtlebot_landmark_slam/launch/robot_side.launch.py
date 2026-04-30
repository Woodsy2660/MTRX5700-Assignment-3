from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    turtlebot3_bringup_dir = get_package_share_directory("turtlebot3_bringup")

    return LaunchDescription([
        # TurtleBot3 hardware bringup (motors, lidar, etc.)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(turtlebot3_bringup_dir, "launch", "robot.launch.py")
            ),
        ),

        # USB camera via v4l2_camera — namespace gives /camera/image_raw
        Node(
            package="v4l2_camera",
            executable="v4l2_camera_node",
            name="camera",
            namespace="camera",
            output="screen",
        ),

        # ArUco landmark detector — subscribes to /camera/image_raw
        # and /pointcloud2d, publishes /landmarks
        Node(
            package="turtlebot_landmark_slam",
            executable="landmark_publisher_node.py",
            name="landmark_publisher",
            output="screen",
        ),
    ])
