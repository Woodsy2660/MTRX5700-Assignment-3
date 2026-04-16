"""
Interactive tuner for the range-adaptive clustering distance threshold.
Loads the first scan from a ROS2 bag and lets you drag a slider to find
the right distance_threshold value for extract_circular_objects.

Usage:
    python tune_clustering.py <bag_path> [--topic /scan] [--max-range 5.0]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan, PointCloud

from landmarks_circle_detector import cluster_points

COLORS = plt.cm.tab10.colors


def draw(ax, scan_points, clusters, threshold):
    ax.clear()
    ax.set_aspect("equal")
    ax.set_xlabel("Y (meters)")
    ax.set_ylabel("X (meters)")
    ax.set_title(f"Clustering  |  distance_threshold={threshold:.3f}  |  {len(clusters)} clusters")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.invert_xaxis()

    ax.plot(scan_points[:, 1], scan_points[:, 0], ".", color="lightgray", markersize=3, zorder=1)

    for i, c in enumerate(clusters):
        color = COLORS[i % len(COLORS)]
        ax.plot(c[:, 1], c[:, 0], ".", color=color, markersize=6, zorder=2)

    ax.plot(0, 0, "^", color="black", markersize=10, zorder=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune clustering distance threshold interactively.")
    parser.add_argument("bag_path", help="Path to ROS2 bag directory")
    parser.add_argument("--topic", default="/scan", help="Scan topic (LaserScan or PointCloud)")
    parser.add_argument("--max-range", type=float, default=None, help="Max range filter (meters)")
    args = parser.parse_args()

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=args.bag_path, storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topic_type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_type_str = topic_type_map.get(args.topic, "")
    msg_class = PointCloud if "PointCloud" in msg_type_str else LaserScan

    scan_points = None
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != args.topic:
            continue
        msg = deserialize_message(data, msg_class)
        if msg_class is LaserScan:
            angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
            ranges = np.array(msg.ranges, dtype=float)
            valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
            scan_points = np.column_stack([ranges[valid] * np.cos(angles[valid]),
                                           ranges[valid] * np.sin(angles[valid])])
        else:
            scan_points = np.array([[p.x, p.y] for p in msg.points], dtype=float)
        break

    if scan_points is None or len(scan_points) == 0:
        print("No scan found on topic. Check bag path and topic name.")
        exit(1)

    if args.max_range is not None:
        scan_points = scan_points[np.linalg.norm(scan_points, axis=1) <= args.max_range]

    INIT_THRESHOLD = 0.1
    fig, ax = plt.subplots(figsize=(10, 9))
    plt.subplots_adjust(bottom=0.12)

    clusters = cluster_points(scan_points, INIT_THRESHOLD)
    draw(ax, scan_points, clusters, INIT_THRESHOLD)

    ax_slider = plt.axes([0.15, 0.04, 0.7, 0.03])
    slider = Slider(ax_slider, "distance_threshold", 0.01, 1.0, valinit=INIT_THRESHOLD, valstep=0.005)

    def on_change(val):
        clusters = cluster_points(scan_points, slider.val)
        draw(ax, scan_points, clusters, slider.val)
        if args.max_range is not None:
            ax.set_xlim(args.max_range, -args.max_range)
            ax.set_ylim(-args.max_range, args.max_range)
        fig.canvas.draw_idle()

    slider.on_changed(on_change)

    if args.max_range is not None:
        ax.set_xlim(args.max_range, -args.max_range)
        ax.set_ylim(-args.max_range, args.max_range)

    plt.show()
