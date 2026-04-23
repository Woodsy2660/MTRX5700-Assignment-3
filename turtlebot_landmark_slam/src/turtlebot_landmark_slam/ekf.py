import numpy as np
from turtlebot_landmark_slam.types import LandmarkMeasurement, ControlMeasurement
import turtlebot_landmark_slam.utils as utils
from copy import deepcopy


class ExtendedKalmanFilter(object):
    """EKF-SLAM filter that jointly estimates robot pose and landmark positions.

    The state vector has shape (3 + 2*M, 1) where M is the number of observed
    landmarks:  [x, y, yaw, lmk0_x, lmk0_y, lmk1_x, lmk1_y, ...]^T
    """

    def __init__(self) -> None:
        # State vector — grows as new landmarks are discovered (shape N x 1)
        self._state_vector = np.array([[0.0], [0.0], [0.0]])

        sigma_position = np.sqrt(10 ** (-3))
        sigma_orientation = np.sqrt(10 ** (-3))

        # Initial 3x3 robot-pose covariance block (grows to N x N with landmarks)
        self._state_covariance = np.array(
            [
                [sigma_position**2, 0.0, 0.0],
                [0.0, sigma_position**2, 0.0],
                [0.0, 0.0, sigma_orientation**2],
            ]
        )

        # Maps landmark label -> starting row index in the state vector
        self._landmark_index = {}

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def x(self):
        return deepcopy(self._state_vector[0])

    @property
    def y(self):
        return deepcopy(self._state_vector[1])

    @property
    def yaw(self):
        return deepcopy(self._state_vector[2])

    @property
    def pose(self):
        """Robot pose as a (3,) array [x, y, yaw]."""
        return np.array([self.x, self.y, self.yaw], copy=True)

    @property
    def pose_covariance(self):
        """3x3 covariance block for the robot pose."""
        return np.array(self._state_covariance[0:3, 0:3], copy=True)

    @property
    def state_mean(self):
        """Full state vector (robot pose + landmark positions), shape (N, 1)."""
        return np.array(self._state_vector, copy=True)

    @property
    def state_covariance(self):
        """Full N x N state covariance matrix."""
        return np.array(self._state_covariance, copy=True)

    # ------------------------------------------------------------------
    # EKF predict step
    # ------------------------------------------------------------------

    def predict(self, control_meaurement: ControlMeasurement):
        """
        Kalman prediction step.

        Math (block form, with landmarks in the state):
            μ̄_x  = f(μ_x, u)         ← nonlinear motion applied to pose only
            Σ̄_xx = F Σ_xx Fᵀ + W M Wᵀ   ← pose block propagates with Jacobians
            Σ̄_xl = F Σ_xl              ← cross-block: pose motion shifts correlations
            Σ̄_ll = Σ_ll                ← landmark block unchanged (landmarks don't move)

        Where F = ∂f/∂x (pose Jacobian), W = ∂f/∂u (control Jacobian),
        M is the control noise covariance from the odometry measurement.
        """
        # Unpack the incoming odometry measurement: the commanded motion
        # vector u = [dx, dy, dθ]ᵀ in the robot's body frame, plus its
        # covariance M characterising how noisy the wheels/IMU are.
        motion_command = control_meaurement.motion_vector
        motion_covariance = control_meaurement.covariance

        # Grab the current pose from the state. This is a view into the
        # underlying array, don't mutate it in place.
        pose = self._state_vector[0:3]

        # Call the motion model. Returns three things:
        #   predicted_robot_pose = f(pose, u) — the nonlinear pose update
        #   F = ∂f/∂x  (3×3) — how a small change in pose perturbs the output
        #   W = ∂f/∂u  (3×3) — how a small change in control perturbs the output
        # We need F and W to propagate uncertainty; we need predicted_robot_pose
        # to update the mean. The nonlinear model is applied to the mean directly
        # (no Jacobian approximation there); linearisation only enters the
        # covariance propagation below.
        predicted_robot_pose, F, W = utils.Relative2AbsolutePose(pose, motion_command)

        # Write the new pose back into the state vector's top 3 rows.
        # np.copyto writes into the existing memory rather than rebinding
        # important because _state_vector may have landmark rows below that
        # we don't want to disturb.
        np.copyto(self._state_vector[0:3], predicted_robot_pose)

        # Block-wise covariance propagation.
        # Σ is laid out as [[Σ_xx, Σ_xl], [Σ_lx, Σ_ll]] where _xx is 3×3,
        # _xl is 3×(2m), _ll is (2m)×(2m). During prediction only the first
        # row and column of blocks change — landmarks don't move, so Σ_ll
        # stays put.
        Pxx = self._state_covariance[:3, :3]      # pose-pose block
        Pxl = self._state_covariance[:3, 3:]      # pose-landmark cross block (may be empty)

        # Pose block: standard Kalman covariance propagation.
        #   F Σ_xx Fᵀ  — transforms existing pose uncertainty through dynamics.
        #                This term alone makes the ellipse grow and rotate because
        #                heading uncertainty leaks into x-y through the -sin/cos
        #                entries in F.
        #   W M Wᵀ     — injects new uncertainty from noisy control, rotated
        #                from the robot's body frame into the world frame by W.
        new_Pxx = F @ Pxx @ F.T + W @ motion_covariance @ W.T

        # Cross block: this is the piece that the original scaffold forgot.
        # The full-state Jacobian F_full is block-diagonal [[F, 0], [0, I]]
        # because landmarks don't move. When you compute F_full Σ F_fullᵀ, the
        # cross block comes out as F Σ_xl. Skipping this makes the gain at the
        # next update step work with stale correlations — the filter thinks
        # the pose and landmarks are less coupled than they really are, so
        # observation-driven corrections are too small.
        new_Pxl = F @ Pxl

        # Write updated blocks back. Σ must stay symmetric, so both the
        # upper-right (_xl) and lower-left (_lx) cross blocks are assigned.
        self._state_covariance[:3, :3] = new_Pxx
        self._state_covariance[:3, 3:] = new_Pxl
        self._state_covariance[3:, :3] = new_Pxl.T
        # self._state_covariance[3:, 3:] (Σ_ll) deliberately untouched.

    # ------------------------------------------------------------------
    # EKF update step
    # ------------------------------------------------------------------

    def extract_landmark_from_state(self, label, state_mean):
        """Look up a landmark's [x, y] from the state vector by label."""
        if label not in self._landmark_index:
            return None
        idx = self._landmark_index[label]
        return np.array([state_mean[idx][0], state_mean[idx + 1][0]])

    def update(self, landmark_measurement: LandmarkMeasurement, is_new: bool):
        """Correct the state estimate using a single landmark measurement.

        If `is_new` is True the landmark is appended to the state vector and
        the covariance matrix is augmented before the standard EKF update.
        """
        pose = self.pose
        state_covariance = self.state_covariance
        x = self.state_mean

        # ----- STAGE 1: register new landmark if needed -----
        if is_new:
            label = landmark_measurement.label
            print(f"Gotten new landmark {label}")

            # Inverse sensor model: relative measurement → absolute landmark position
            landmark_measured_abs, H1, H2 = utils.Relative2AbsoluteXY(
                pose, [landmark_measurement.x, landmark_measurement.y]
            )

            # Record where in the state vector this landmark's rows will live
            index = x.shape[0]
            print(f"Insertion index {index}")
            self._landmark_index[label] = index

            # Append the landmark's absolute position to the state mean
            x = np.vstack((x, landmark_measured_abs))

            # Build the new landmark's covariance blocks
            Prr = self.pose_covariance

            # Cross-correlation with the rest of the state
            if len(self._landmark_index) == 1:
                # First landmark: cross block is just H1 · Σ_xx
                Plx = np.dot(H1, Prr)
            else:
                # Subsequent landmarks: also co-varies with existing landmarks
                Prm = state_covariance[0:3, 3:]
                Plx = np.dot(H1, np.bmat([[Prr, Prm]]))

            # Self-covariance of the new landmark
            Pll = (np.dot(H1, np.dot(Prr, H1.T))
                   + np.dot(H2, np.dot(landmark_measurement.covariance, H2.T)))

            # Extend Σ by one block-row and one block-column
            P = np.bmat([[state_covariance, Plx.T], [Plx, Pll]])
            state_covariance = np.array(P, copy=True)

            # Commit augmented state so the Kalman update below sees it
            self._state_vector = np.array(x, copy=True)
            self._state_covariance = np.array(state_covariance, copy=True)

        # ----- STAGE 2: Kalman update for this landmark -----
        label = landmark_measurement.label
        estimated_landmark = self.extract_landmark_from_state(label, x)

        # Sensor model: expected measurement h(μ) and Jacobians
        expected_measurement, Hr, Hl = utils.Absolute2RelativeXY(pose, estimated_landmark)

        # Actual measurement z
        Z = np.array([[landmark_measurement.x], [landmark_measurement.y]])

        # Build the sparse C matrix (2 × N)
        index = self._landmark_index[label]
        C = np.zeros((2, state_covariance.shape[0]))
        C[:, :3] = Hr
        C[:, index:index + 2] = Hl

        R = landmark_measurement.covariance

        # Innovation
        y = Z - expected_measurement

        # Innovation covariance
        S = C @ state_covariance @ C.T + R

        # Guard against near-singular S
        if np.linalg.det(S) < 1e-6:
            S = ExtendedKalmanFilter.reguarlise_matrix(S)

        # Kalman gain
        K = state_covariance @ C.T @ np.linalg.inv(S)

        # State correction
        posterior_state_mean = x + K @ y

        # Covariance contraction
        I = np.eye(len(posterior_state_mean))
        posterior_state_covariance = (I - K @ C) @ state_covariance

        # Commit posterior
        self._state_vector = np.array(posterior_state_mean, copy=True)
        self._state_covariance = np.array(posterior_state_covariance, copy=True)
