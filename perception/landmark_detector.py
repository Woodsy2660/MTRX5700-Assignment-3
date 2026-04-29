# Core landmark detection logic - used by both the offline test script and the ROS2 node
# No ROS dependencies here so it runs on Mac for testing

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class LandmarkObservation:
    tag_id:   int
    bearing:  float   # radians, positive = right of robot heading
    range_m:  float   # metres, -1.0 if no lidar match
    is_new:   bool
    pixel_cx: float
    pixel_cy: float
    x:        float = 0.0   # robot body frame, x = forward
    y:        float = 0.0   # robot body frame, y = left
    s_x:      float = 0.0   # std dev of x measurement
    s_y:      float = 0.0   # std dev of y measurement


class LandmarkDetector:

    # TODO: replace with calibrated values from Assignment 2
    DEFAULT_FX = 530.0   # focal length in pixels
    DEFAULT_CX = 320.0   # principal point x

    # markers smaller than this are almost certainly false positives
    # checked against bag data: real tags >= 22px, false positive was 11px
    MIN_MARKER_SIDE_PX = 20

    # how close a lidar point needs to be (in angle) to count as a match
    LIDAR_ANGLE_TOL_RAD = np.deg2rad(5.0)

    # sensor noise estimates for covariance propagation
    # TODO: update s_bearing from actual calibration error analysis
    SIGMA_RANGE_M     = 0.02               # RPLIDAR A1 spec
    SIGMA_BEARING_RAD = np.deg2rad(1.5)

    def __init__(self, fx: float = DEFAULT_FX, cx: float = DEFAULT_CX):
        self.fx = fx
        self.cx = cx

        # try 4x4 first since that's what the lab markers use
        self._detectors = []
        for dict_id in [
            cv2.aruco.DICT_4X4_50,
            cv2.aruco.DICT_5X5_100,
            cv2.aruco.DICT_6X6_250,
            cv2.aruco.DICT_4X4_100,
        ]:
            d = cv2.aruco.getPredefinedDictionary(dict_id)
            p = cv2.aruco.DetectorParameters()
            self._detectors.append(cv2.aruco.ArucoDetector(d, p))

        # tracks which landmarks have been seen before across frames
        self.registry: dict[int, dict] = {}

    def process_image(
        self,
        img_bgr: np.ndarray,
        timestamp_s: float,
        pointcloud_xy: np.ndarray | None = None,
    ) -> list[LandmarkObservation]:
        # pointcloud_xy is Nx2 [x, y] in robot frame (x forward, y left)
        # pass None if you don't have lidar yet
        img_h, img_w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        observations = []

        for detector in self._detectors:
            corners_list, ids, _ = detector.detectMarkers(gray)
            if ids is None or len(ids) == 0:
                continue

            for tag_corners, tag_id in zip(corners_list, ids.flatten().tolist()):
                pts = tag_corners[0]

                if not self._is_valid(pts):
                    continue

                cx_px = float(np.mean(pts[:, 0]))
                cy_px = float(np.mean(pts[:, 1]))
                bearing = self._bearing(cx_px)

                range_m = -1.0
                if pointcloud_xy is not None and len(pointcloud_xy) > 0:
                    range_m = self._range_from_lidar(pointcloud_xy, bearing)

                is_new = tag_id not in self.registry
                self._update_registry(tag_id, timestamp_s)

                x, y, s_x, s_y = self._to_cartesian(range_m, bearing)

                observations.append(LandmarkObservation(
                    tag_id=tag_id, bearing=bearing, range_m=range_m,
                    is_new=is_new, pixel_cx=cx_px, pixel_cy=cy_px,
                    x=x, y=y, s_x=s_x, s_y=s_y,
                ))

            if observations:
                break  # stop once a dictionary gives us something valid

        return observations

    def annotate(self, img_bgr: np.ndarray, observations: list[LandmarkObservation]) -> np.ndarray:
        out = img_bgr.copy()
        img_h = out.shape[0]

        for obs in observations:
            colour = (0, 255, 0) if obs.is_new else (0, 200, 255)
            status = "NEW" if obs.is_new else f"#{self.registry[obs.tag_id]['count']}x"
            label  = f"id={obs.tag_id} {status}"
            if obs.range_m > 0:
                label += f"  r={obs.range_m:.2f}m"
            label += f"  {np.rad2deg(obs.bearing):+.1f}deg"

            cv2.putText(out, label,
                        (max(int(obs.pixel_cx) - 60, 0), max(int(obs.pixel_cy) - 15, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
            cv2.circle(out, (int(obs.pixel_cx), int(obs.pixel_cy)), 5, colour, -1)

        known = sorted(self.registry.keys())
        cv2.putText(out, f"Known landmarks: {known}",
                    (10, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return out

    def _is_valid(self, pts: np.ndarray) -> bool:
        side = float(np.linalg.norm(pts[0] - pts[1]))
        return side >= self.MIN_MARKER_SIDE_PX

    def _bearing(self, px: float) -> float:
        # atan2 gives accurate bearing using focal length rather than linear approx
        return float(np.arctan2(px - self.cx, self.fx))

    def _range_from_lidar(self, pointcloud_xy: np.ndarray, bearing_rad: float) -> float:
        # camera positive = right, ROS y positive = left, so lidar angle = -bearing
        lidar_angle = -bearing_rad
        angles = np.arctan2(pointcloud_xy[:, 1], pointcloud_xy[:, 0])
        diffs  = np.abs(angles - lidar_angle)
        diffs  = np.minimum(diffs, 2 * np.pi - diffs)  # handle wrap at +-pi

        idx = int(np.argmin(diffs))
        if diffs[idx] > self.LIDAR_ANGLE_TOL_RAD:
            return -1.0

        x, y = pointcloud_xy[idx]
        return float(np.sqrt(x * x + y * y))

    def _to_cartesian(self, range_m: float, bearing_rad: float) -> tuple[float, float, float, float]:
        # convert range-bearing to robot body frame x, y
        # then propagate sensor uncertainty through the Jacobian to get s_x, s_y
        #
        # x =  r * cos(theta)
        # y = -r * sin(theta)   (negated because bearing+ = right = -y in ROS)
        #
        # Jacobian d(x,y)/d(r,theta):
        #   J = [[ cos(t), -r*sin(t)],
        #        [-sin(t), -r*cos(t)]]
        #
        # Sigma_xy = J * diag(sr^2, st^2) * J^T
        #   s_x = sqrt(cos^2(t)*sr^2 + r^2*sin^2(t)*st^2)
        #   s_y = sqrt(sin^2(t)*sr^2 + r^2*cos^2(t)*st^2)

        if range_m <= 0:
            return 0.0, 0.0, 0.0, 0.0

        r, t  = range_m, bearing_rad
        x     =  r * np.cos(t)
        y     = -r * np.sin(t)
        sr, st = self.SIGMA_RANGE_M, self.SIGMA_BEARING_RAD
        s_x   = float(np.sqrt((np.cos(t) * sr) ** 2 + (r * np.sin(t) * st) ** 2))
        s_y   = float(np.sqrt((np.sin(t) * sr) ** 2 + (r * np.cos(t) * st) ** 2))

        return float(x), float(y), s_x, s_y

    def _update_registry(self, tag_id: int, timestamp_s: float) -> None:
        if tag_id not in self.registry:
            self.registry[tag_id] = {"first_t": timestamp_s, "last_t": timestamp_s, "count": 1}
        else:
            self.registry[tag_id]["last_t"] = timestamp_s
            self.registry[tag_id]["count"] += 1
