from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # /odom (Odometry) -> /control (Twist), reused from sim mode
        Node(
            package="turtlebot_landmark_slam",
            executable="odom_to_control_republisher.py",
            name="odom_to_control",
            output="screen",
            parameters=[{"use_sim_time": True}],
        ),

        # EKF in real mode, with init pose remapped to bag's /odom
        Node(
            package="turtlebot_landmark_slam",
            executable="ekf_pipeline_node.py",
            name="ekf",
            output="screen",
            parameters=[{
                "is_real": True,
                "use_sim_time": True,
            }],
        ),

        # ArUco landmark detector — subscribes to /camera/image_raw
        # and /pointcloud2d from the bag, publishes /landmarks (body-frame x,y)
        Node(
            package="turtlebot_landmark_slam",
            executable="landmark_publisher_node.py",
            name="landmark_publisher",
            output="screen",
            parameters=[{"use_sim_time": True}],
        ),

        # Trajectory line publisher
        Node(
            package="turtlebot_landmark_slam",
            executable="path_publisher.py",
            name="path_publisher",
            output="screen",
            parameters=[{"use_sim_time": True}],
        ),
    ])