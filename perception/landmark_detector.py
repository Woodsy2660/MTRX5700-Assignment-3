"""
LandmarkDetector — core logic shared between offline testing and the ROS2 node.

This file has zero ROS2 dependencies. It takes a BGR image (numpy array) and
optionally a lidar point cloud (Nx2 numpy array of [x, y] in robot frame), and
returns a list of LandmarkObservation objects ready to feed into the EKF.

Used by:
  inspect_bag.py              — offline, reads .mcap directly on Mac
  landmark_publisher_node.py  — ROS2 node on the Linux laptop
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import cv2


# ── Observation dataclass ─────────────────────────────────────────────────────
# One of these is produced per detected tag per frame.
# This is what gets packaged into a LandmarkMsg and sent to the EKF.

@dataclass
class LandmarkObservation:
    tag_id:    int
    bearing:   float        # radians, robot frame — positive = right of heading
    range_m:   float        # metres from lidar; -1.0 if lidar not available yet
    is_new:    bool         # True = first time this tag_id has been seen
    pixel_cx:  float        # tag centre x in image (for debug visualisation)
    pixel_cy:  float        # tag centre y in image (for debug visualisation)


# ── Detector class ────────────────────────────────────────────────────────────

class LandmarkDetector:
    # Camera — replace with values from your Assignment 2 calibration
    # fx: focal length in pixels (used for accurate bearing via atan2)
    # cx: principal point x (usually ≈ image_width / 2)
    DEFAULT_FX  = 530.0     # pixels — typical RPi cam at 640px wide
    DEFAULT_CX  = 320.0     # pixels

    # False-positive rejection threshold
    # Real cylinder tags in this bag: side ≥ 22px. ID 17 false positive: 11px.
    # A threshold of 20px cleanly separates them with headroom on both sides.
    MIN_MARKER_SIDE_PX = 20

    # Lidar bearing-match tolerance
    LIDAR_ANGLE_TOL_RAD = np.deg2rad(5.0)  # ±5° window around tag bearing

    def __init__(self, fx: float = DEFAULT_FX, cx: float = DEFAULT_CX):
        self.fx = fx
        self.cx = cx

        # Build ArUco detectors — try 4×4 first (what the lab uses), fall back
        self._detectors: list[cv2.aruco.ArucoDetector] = []
        for dict_id in [
            cv2.aruco.DICT_4X4_50,
            cv2.aruco.DICT_5X5_100,
            cv2.aruco.DICT_6X6_250,
            cv2.aruco.DICT_4X4_100,
        ]:
            d = cv2.aruco.getPredefinedDictionary(dict_id)
            p = cv2.aruco.DetectorParameters()
            self._detectors.append(cv2.aruco.ArucoDetector(d, p))

        # Landmark registry — persists across frames (and across ROS2 callbacks)
        # registry[tag_id] = { first_t, last_t, count }
        self.registry: dict[int, dict] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def process_image(
        self,
        img_bgr: np.ndarray,
        timestamp_s: float,
        pointcloud_xy: np.ndarray | None = None,
    ) -> list[LandmarkObservation]:
        """
        Detect ArUco tags in img_bgr, apply false-positive filters, compute
        bearing (always) and range (if pointcloud_xy provided).

        pointcloud_xy: Nx2 float32 array of lidar points in robot frame [x, y]
                       where x = forward, y = left (standard ROS convention).
                       Pass None if you haven't added lidar yet.

        Returns a list of LandmarkObservation — one per valid detection.
        """
        img_h, img_w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        observations: list[LandmarkObservation] = []

        for detector in self._detectors:
            corners_list, ids, _ = detector.detectMarkers(gray)
            if ids is None or len(ids) == 0:
                continue

            for tag_corners, tag_id in zip(corners_list, ids.flatten().tolist()):
                pts = tag_corners[0]  # shape (4, 2)

                if not self._is_valid(pts, img_h, img_w):
                    continue

                cx_px = float(np.mean(pts[:, 0]))
                cy_px = float(np.mean(pts[:, 1]))
                bearing = self._bearing(cx_px)

                range_m = -1.0
                if pointcloud_xy is not None and len(pointcloud_xy) > 0:
                    range_m = self._range_from_lidar(pointcloud_xy, bearing)

                is_new = tag_id not in self.registry
                self._update_registry(tag_id, timestamp_s)

                observations.append(LandmarkObservation(
                    tag_id   = tag_id,
                    bearing  = bearing,
                    range_m  = range_m,
                    is_new   = is_new,
                    pixel_cx = cx_px,
                    pixel_cy = cy_px,
                ))

            # Stop after the first dictionary that returns any valid detections
            if observations:
                break

        return observations

    def annotate(
        self,
        img_bgr: np.ndarray,
        observations: list[LandmarkObservation],
    ) -> np.ndarray:
        """Draw detection results onto a copy of the image for visualisation."""
        out = img_bgr.copy()
        img_h, img_w = out.shape[:2]

        for obs in observations:
            colour = (0, 255, 0) if obs.is_new else (0, 200, 255)
            status = "NEW" if obs.is_new else f"#{self.registry[obs.tag_id]['count']}x"
            bearing_deg = np.rad2deg(obs.bearing)

            label = f"id={obs.tag_id} {status}"
            if obs.range_m > 0:
                label += f"  r={obs.range_m:.2f}m"
            label += f"  {bearing_deg:+.1f}deg"

            x = int(obs.pixel_cx)
            y = max(int(obs.pixel_cy) - 15, 20)
            cv2.putText(out, label, (x - 60, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
            cv2.circle(out, (int(obs.pixel_cx), int(obs.pixel_cy)), 5, colour, -1)

        known = sorted(self.registry.keys())
        cv2.putText(out, f"Known landmarks: {known}",
                    (10, img_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return out

    # ── Private helpers ───────────────────────────────────────────────────────

    def _is_valid(self, pts: np.ndarray, img_h: int, img_w: int) -> bool:
        """Return False if the detection looks like a false positive.

        Real cylinder tags in this dataset have side ≥ 22px.
        The only confirmed false positive (ID 17, ceiling reflection) was 11px.
        Size is the reliable separator — vertical position is not, because
        real tags appear anywhere from cy/h ≈ 0.22 to 0.44 depending on range.
        """
        side = float(np.linalg.norm(pts[0] - pts[1]))
        return side >= self.MIN_MARKER_SIDE_PX

    def _bearing(self, px: float) -> float:
        """
        Horizontal bearing in radians using camera intrinsics.
        bearing = atan2(px - cx, fx)
        Positive = right of heading, negative = left.
        Replace fx and cx with your Assignment 2 calibration values.
        """
        return float(np.arctan2(px - self.cx, self.fx))

    def _range_from_lidar(
        self, pointcloud_xy: np.ndarray, bearing_rad: float
    ) -> float:
        """
        Find the closest lidar point to the camera bearing direction.

        pointcloud_xy: Nx2 array, robot frame (x=forward, y=left).
        bearing_rad: positive = right — camera convention.

        Camera "right" = robot "-y", so lidar angle = -bearing.
        Returns range in metres, or -1.0 if no point falls within tolerance.
        """
        lidar_angle = -bearing_rad  # convert to ROS robot-frame convention

        # atan2(y, x) gives angle of each point in robot frame
        angles = np.arctan2(pointcloud_xy[:, 1], pointcloud_xy[:, 0])
        diffs  = np.abs(angles - lidar_angle)

        # Handle wrap-around at ±π
        diffs = np.minimum(diffs, 2 * np.pi - diffs)

        closest_idx = int(np.argmin(diffs))
        if diffs[closest_idx] > self.LIDAR_ANGLE_TOL_RAD:
            return -1.0  # no lidar point near this bearing

        x, y = pointcloud_xy[closest_idx]
        return float(np.sqrt(x * x + y * y))

    def _update_registry(self, tag_id: int, timestamp_s: float) -> None:
        if tag_id not in self.registry:
            self.registry[tag_id] = {
                "first_t": timestamp_s,
                "last_t":  timestamp_s,
                "count":   1,
            }
        else:
            self.registry[tag_id]["last_t"] = timestamp_s
            self.registry[tag_id]["count"] += 1
