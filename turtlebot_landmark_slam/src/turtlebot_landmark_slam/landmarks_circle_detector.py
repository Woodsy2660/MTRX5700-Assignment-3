import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as pltCircle
from dataclasses import dataclass


@dataclass
class CircleFit:
    """
    Data class to hold the results of a circle fit, including the original points, fitted parameters, fit quality metrics, and covariance of the estimates.
    """

    # Original points belonging to the cluster that was fitted, as an Nx2 array of (x, y) in meters.
    points: np.ndarray
    # Center of the fitted circle in meters, as (cx, cy) in Cartesian or (range, bearing) if polar=True
    center: tuple[float, float]
    # Radius of the fitted circle in meters
    radius: float
    # Mean Squared Error of the geometric fit (average squared distance from points to the circle), in meters^2.
    mse: float
    # If polar=True, center is (range, bearing) and covariance rows/cols 0-1 correspond to (range, bearing) rather than (x, y).
    polar: bool
    # 3x3 covariance matrix for (cx, cy, radius) or (range, bearing, radius) if polar=True
    covariance: np.ndarray


def cluster_points(scan_points, distance_threshold):
    """
    Splits ordered lidar scan points into clusters using a range-adaptive distance threshold.

    threshold(r) = r * distance_threshold, so the effective angular gap is constant
    regardless of range. Handles the circular wrap-around of the scan (last point is
    angularly adjacent to the first).

    Args:
        scan_points: Nx2 array of (x, y) points in scan order.
        distance_threshold: Max gap at 1 m range to stay in the same cluster.

    Returns:
        List of Nx2 numpy arrays, one per cluster.
    """
    dist_diffs = np.linalg.norm(scan_points[1:] - scan_points[:-1], axis=1)
    point_ranges = np.linalg.norm(scan_points[:-1], axis=1)
    split_indices = np.where(dist_diffs >= point_ranges * distance_threshold)[0] + 1
    clusters = list(np.split(scan_points, split_indices))

    # Wrap-around check: the scan is circular so the last point is angularly
    # adjacent to the first. Merge the first and last clusters if close enough.
    if len(clusters) > 1:
        wrap_gap = np.linalg.norm(scan_points[-1] - scan_points[0])
        wrap_threshold = np.linalg.norm(scan_points[-1]) * distance_threshold
        if wrap_gap < wrap_threshold:
            clusters[0] = np.vstack([clusters[-1], clusters[0]])
            clusters.pop()

    return clusters


def extract_circular_objects(
    scan_points,
    distance_threshold=0.05,
    min_points=4,
    max_radius=0.12,
    min_radius=0.06,
    max_mse=1.0e-5,
    max_aspect_ratio=None,
    min_arc_angle=None,
    min_center_range=None,
    polar=False,
):
    """
    Groups ordered laser scan points into clusters and fits circles to each.

    Args:
        scan_points: Nx2 numpy array of (x, y) coordinates from a sequential scan.
        distance_threshold: Max gap between consecutive points at 1 m range to remain in the
                            same cluster. Scales linearly with range: threshold(r) = r * distance_threshold.
                            This matches the angular spacing of a lidar beam, so the threshold adapts
                            to the sparser point density at longer ranges.
        min_points: Minimum points required to attempt a circle fit.
        max_radius: Maximum allowable radius (meters) to filter out straight walls/lines.
        min_radius: Minimum allowable radius (meters). Rejects spuriously small fits.
        max_mse: If set, rejects fits whose MSE exceeds this value (meters^2).
        max_aspect_ratio: If set, rejects clusters whose PCA eigenvalue ratio (largest/smallest)
                          exceeds this before fitting. Walls and corners are elongated (high ratio);
                          circular arcs are compact. Try 4.0–8.0. Saves fitting cost on bad clusters.
        min_arc_angle: If set (radians), rejects fits where the cluster spans less than this angle
                       around the fitted center. Corners cover ~90° or less; try np.radians(60)
                       to require at least 60° of arc coverage.
        min_center_range: If set (meters), rejects fits whose center is closer than this distance
                          to the sensor origin. Filters robot body reflections and near-field noise.
        polar: If True, center is returned as (range, bearing) in metres/radians instead of (x, y).
               The covariance is also transformed accordingly.

    Returns:
        List of CircleFit containing fit parameters and uncertainty metrics.
        When polar=True, CircleFit.center is (range, bearing) and covariance rows/cols
        0-1 correspond to (range, bearing) rather than (x, y).
    """
    if len(scan_points) < min_points:
        return []

    # 1. Cluster scan points using range-adaptive distance threshold
    clusters = cluster_points(scan_points, distance_threshold)

    # Keep only clusters with enough points
    valid_clusters = [c for c in clusters if len(c) >= min_points]

    # 2. Fitting circles to each valid cluster
    results = []
    for cluster in valid_clusters:
        # Pre-fit: reject elongated clusters (walls, corners) via PCA aspect ratio.
        # A circular arc has balanced spread; walls/corners are strongly elongated.
        if max_aspect_ratio is not None:
            # aspect ratio is a measure of how elongated the cluster is. A perfect circle has an aspect ratio of 1.
            # a high aspect ratio indicates a cluster that is more like a line (wall) than a circle.
            # By filtering out clusters with a high aspect ratio, we can focus our circle fitting on clusters that are
            # more likely to represent circular features in the environment.
            eigenvalues = np.linalg.eigvalsh(np.cov(cluster.T))
            if eigenvalues[0] < 1e-9:
                # Avoid division by zero for perfectly linear clusters
                continue
            if eigenvalues[-1] / eigenvalues[0] > max_aspect_ratio:
                continue

        # Fit a circle to the cluster and estimate its covariance
        fit = fit_circle_with_covariance(cluster, max_radius)
        if fit is None:
            continue

        # Apply additional filters based on fit quality and geometry
        if fit.radius < min_radius:
            continue

        if max_mse is not None and fit.mse > max_mse:
            continue

        # Reject fits whose centers are too close to the sensor origin,
        # which are likely due to noise or reflections from the robot itself.
        cx, cy = fit.center
        if min_center_range is not None and np.sqrt(cx**2 + cy**2) < min_center_range:
            continue

        # Reject fits where the cluster points only cover a small arc around the center,
        # which are likely to be corners or partial walls rather than true circular features.
        if min_arc_angle is not None:
            angles = np.arctan2(cluster[:, 1] - cy, cluster[:, 0] - cx)
            arc_span = np.ptp(np.unwrap(angles))
            if arc_span < min_arc_angle:
                continue

        # Convert to polar coordinates, recompute covariance, and propagate uncertainty
        if polar:
            fit = _to_polar(fit)

        results.append(fit)

    return results


def _to_polar(fit: "CircleFit") -> "CircleFit":
    """Convert a CircleFit's center from (x, y) to (range, bearing) and propagate covariance."""
    cx, cy = fit.center
    rng = np.sqrt(cx**2 + cy**2)
    bearing = np.arctan2(cy, cx)

    # Jacobian of [range, bearing] w.r.t. [cx, cy]
    # d(range)/d(cx)   = cx / rng
    # d(range)/d(cy)   = cy / rng
    # d(bearing)/d(cx) = -cy / rng^2
    # d(bearing)/d(cy) =  cx / rng^2
    J = np.array(
        [
            [cx / rng, cy / rng, 0.0],
            [-cy / rng**2, cx / rng**2, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Propagate covariance: cov_polar = J * cov_cartesian * J^T
    cov_polar = J @ fit.covariance @ J.T

    return CircleFit(
        points=fit.points,
        center=(rng, bearing),
        radius=fit.radius,
        mse=fit.mse,
        covariance=cov_polar,
        polar=True,
    )


def fit_circle_with_covariance(points, max_radius=1.5):
    """
    Fits a circle using Algebraic Least Squares for an initial guess,
    followed by Non-Linear Least Squares (Levenberg-Marquardt) for geometric accuracy.
    Estimates the covariance matrix of (cx, cy, r).
    """
    x = points[:, 0]
    y = points[:, 1]
    n = len(points)

    # --- Phase 1: Initial Guess via Algebraic Circle Fit (Kasa Method) ---
    A = np.column_stack([2 * x, 2 * y, np.ones(n)])
    b = x**2 + y**2

    try:
        params_alg, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx_init, cy_init, k = params_alg
        r_sq = k + cx_init**2 + cy_init**2
        if r_sq <= 0:
            return None
        r_init = np.sqrt(r_sq)
    except np.linalg.LinAlgError:
        return None

    # Quick rejection if it's obviously a straight line acting as a huge circle
    if r_init > max_radius * 3:
        return None

    # --- Phase 2: Refinement via Non-Linear Least Squares ---
    # Minimizes the true geometric distance: error = sqrt((x-cx)^2 + (y-cy)^2) - r
    def geometric_residuals(params, x, y):
        cx, cy, r = params
        return np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r

    initial_guess = [cx_init, cy_init, r_init]

    try:
        # Use Levenberg-Marquardt ('lm') which is ideal for this unconstrained optimization
        opt_res = least_squares(
            geometric_residuals, initial_guess, args=(x, y), method="lm"
        )
        cx, cy, r = opt_res.x
    except ValueError:
        return None

    # Final filter to reject flat surfaces
    if r > max_radius or r <= 0.01:
        return None

    # --- Phase 3: Quality Metrics and Covariance ---
    # MSE of the geometric fit
    mse = np.mean(opt_res.fun**2)

    # Estimate Covariance from the Jacobian (J) returned by Levenberg-Marquardt
    # Formula: Cov = sigma^2 * (J^T J)^-1
    J = opt_res.jac
    try:
        # Use pseudo-inverse in case J^T J is near-singular, which can happen with small clusters or nearly collinear points.
        H_inv = np.linalg.inv(J.T @ J)
        # Unbiased variance estimate: Sum of squared residuals / degrees of freedom
        degrees_of_freedom = max(1, n - 3)
        # sigma^2 is the variance of the residuals, which we estimate from the MSE of the fit.
        # This gives us a measure of how much the observed points deviate from the fitted circle,
        # and it scales the covariance to reflect the actual fit quality.
        sigma_sq = np.sum(opt_res.fun**2) / degrees_of_freedom
        covariance = sigma_sq * H_inv
    except np.linalg.LinAlgError:
        # Matrix is singular (e.g., points are perfectly collinear)
        covariance = np.full((3, 3), np.inf)

    return CircleFit(
        points=points,
        center=(cx, cy),
        radius=r,
        mse=mse,
        covariance=covariance,
        polar=False,
    )


if __name__ == "__main__":
    import argparse
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import LaserScan, PointCloud

    parser = argparse.ArgumentParser(
        description="Detect circles from laser scans in a ROS2 bag."
    )
    parser.add_argument("bag_path", help="Path to the ROS2 bag directory")
    parser.add_argument(
        "--topic", default="/scan", help="Scan topic name (LaserScan or PointCloud)"
    )
    parser.add_argument(
        "--max-range",
        type=float,
        default=None,
        help="Maximum range (meters) to include points. Points beyond this are filtered out before circle detection.",
    )
    args = parser.parse_args()

    # Inspect topic type from the bag metadata
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=args.bag_path, storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)

    topic_type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_type_str = topic_type_map.get(args.topic, "")
    print(f"Topic '{args.topic}' has type: {msg_type_str or '(not found)'}")

    if "LaserScan" in msg_type_str:
        msg_class = LaserScan
    elif "PointCloud" in msg_type_str:
        msg_class = PointCloud
    else:
        print(
            f"Unsupported or unknown message type '{msg_type_str}'. Trying LaserScan."
        )
        msg_class = LaserScan

    def laserscan_to_points(msg):
        angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
        ranges = np.array(msg.ranges, dtype=float)
        valid = (
            np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
        )
        return np.column_stack(
            [
                ranges[valid] * np.cos(angles[valid]),
                ranges[valid] * np.sin(angles[valid]),
            ]
        )

    scans = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != args.topic:
            continue
        msg = deserialize_message(data, msg_class)
        if msg_class is LaserScan:
            pts = laserscan_to_points(msg)
        else:
            pts = np.array([[p.x, p.y] for p in msg.points], dtype=float)
        if len(pts) >= 5:
            scans.append(pts)

    print(f"Loaded {len(scans)} scans from '{args.topic}'")
    if not scans:
        print("No scans found. Check the topic name.")
        exit(1)

    COLORS = [
        "tab:red",
        "tab:green",
        "tab:blue",
        "tab:purple",
        "tab:orange",
        "tab:cyan",
        "tab:brown",
        "tab:pink",
        "tab:olive",
        "tab:gray",
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle("Click or press any key to advance to the next scan", fontsize=10)

    for scan_idx, scan_points in enumerate(scans):
        ax.clear()
        ax.set_aspect("equal")
        # Robotic convention: x forward (up), y left (left).
        # Map: plot horizontal = robot Y (inverted), plot vertical = robot X.
        ax.set_xlabel("Y (meters)")
        ax.set_ylabel("X (meters)")
        ax.invert_xaxis()
        ax.set_title(f"Scan {scan_idx + 1} / {len(scans)}")
        ax.grid(True, linestyle=":", alpha=0.6)

        if args.max_range is not None:
            ranges = np.linalg.norm(scan_points, axis=1)
            scan_points = scan_points[ranges <= args.max_range]
            ax.set_xlim(args.max_range, -args.max_range)
            ax.set_ylim(-args.max_range, args.max_range)

        ax.plot(
            scan_points[:, 1],
            scan_points[:, 0],
            ".",
            color="lightgray",
            label="Raw scan",
            markersize=4,
            zorder=2,
        )
        ax.plot(
            0, 0, "^", color="black", markersize=10, label="Sensor origin", zorder=5
        )

        detected = extract_circular_objects(scan_points, polar=True)
        print(f"\nScan {scan_idx + 1}: {len(detected)} circle(s) detected")

        for i, c in enumerate(detected):
            color = COLORS[i % len(COLORS)]
            rng, bearing = c.center
            cx = rng * np.cos(bearing)
            cy = rng * np.sin(bearing)

            ax.plot(
                c.points[:, 1],
                c.points[:, 0],
                ".",
                color=color,
                markersize=8,
                label=f"Circle {i+1}: r={c.radius:.2f}m",
                zorder=3,
            )
            ax.add_patch(
                pltCircle(
                    (cy, cx), c.radius, color=color, fill=False, linewidth=2, zorder=4
                )
            )
            ax.plot(cy, cx, "+", color=color, markersize=10, zorder=5)

            print(
                f"  Circle {i+1}: range={rng:.3f} m, bearing={np.degrees(bearing):.2f} deg, "
                f"radius={c.radius:.3f} m, mse={c.mse:.2e}"
            )

        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
        plt.waitforbuttonpress()

    plt.show()
