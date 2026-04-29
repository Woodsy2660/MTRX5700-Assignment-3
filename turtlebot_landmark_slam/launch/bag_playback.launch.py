from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Declare 'odom' as the root TF frame (bag has no /tf topics)
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="odom_frame_publisher",
            arguments=["0", "0", "0", "0", "0", "0", "1", "map", "odom"],
        ),
        # Convert /cmd_vel (TwistStamped, Jazzy) -> /cmd_vel_twist (Twist)
        Node(
            package="turtlebot_landmark_slam",
            executable="twist_stamped_republisher.py",
            name="twist_stamped_republisher",
            output="screen",
        ),
        # Detect cylindrical landmarks from /pointcloud2d, publish /landmarks
        Node(
            package="turtlebot_landmark_slam",
            executable="landmark_detector_node.py",
            name="landmark_detector",
            output="screen",
        ),
        # EKF SLAM — real-robot mode, control from the republished Twist topic
        Node(
            package="turtlebot_landmark_slam",
            executable="ekf_pipeline_node.py",
            name="ekf",
            output="screen",
            parameters=[{"is_real": True}],
            remappings=[
                ("~/landmarks", "/landmarks"),
                ("~/control", "/cmd_vel_twist"),
            ],
        ),
        # Publish /ekf/path (EKF estimate) and /gt/path (wheel odometry)
        Node(
            package="turtlebot_landmark_slam",
            executable="path_publisher.py",
            name="path_publisher",
            output="screen",
        ),
    ])
