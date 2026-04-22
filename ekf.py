"""
Extended Kalman Filter (EKF) implementation for SLAM.

Tracks a joint state [robot_pose, landmark_1, landmark_2, ...] as a single
Gaussian belief N(μ, Σ). The robot pose is 3D (x, y, θ); each landmark is
2D (x, y). Every prediction and every observation operates on this one
joint distribution, which is what makes the coupling between pose and map
automatic — correcting one corrects everything it's correlated with.
"""
import numpy as np
from copy import deepcopy
from typing import List

from math_utils import utils, se2
from measurements import LandmarkMeasurement, ControlMeasurement


class ExtendedKalmanFilter(object):

    def __init__(self) -> None:
        # State vector μ starts as just the robot pose [x, y, θ]ᵀ at the origin.
        # Landmarks will be appended later as they're discovered the state
        # grows from 3 to 3+2m over time, where m is the number of known landmarks.
        self._state_vector = np.array([[0.0], [0.0], [0.0]])

        # Initial pose uncertainty: 1e-3 m position std, 1e-3 rad orientation std.
        # This represents "I know almost exactly where I am at t=0". In a SLAM
        # system the world frame is usually *defined* to coincide with the initial
        # pose, so this tiny uncertainty is realistic.
        sigma_position = np.sqrt(np.pow(10.0, -3))
        sigma_orientation = np.sqrt(np.pow(10.0, -3))

        # Σ₀: initial covariance. Diagonal → x, y, θ uncorrelated at start.
        # Off-diagonals will grow over time as the filter learns that heading
        # uncertainty couples into position uncertainty (via F Σ Fᵀ in predict).
        self._state_covariance = np.array(
            [
                [np.pow(sigma_position, 2), 0., 0.],
                [0., np.pow(sigma_position, 2), 0.],
                [0., 0., np.pow(sigma_orientation, 2)]
            ]
        )

        # Dictionary mapping landmark ID → row index in the state vector where
        # that landmark's [x, y] pair starts. Without this, we'd have no way
        # to find landmark 7 inside a 23-element state vector.
        self._landmark_index = {}

    # ------------------------------------------------------------------
    # Read-only accessors (deepcopy prevents callers mutating internal state)
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
        # Returns just the pose portion [x, y, θ] — the first 3 state components.
        return np.array([self.x, self.y, self.yaw], copy=True)

    @property
    def pose_covariance(self):
        # Top-left 3×3 block of Σ — the robot's own uncertainty, ignoring
        # correlations with landmarks.
        return np.array(self._state_covariance[0:3, 0:3], copy=True)

    @property
    def state_mean(self):
        # The whole joint mean μ.
        return np.array(self._state_vector, copy=True)

    @property
    def num_landmarks(self):
        # State is [pose (3), l1 (2), l2 (2), ...]. Rows beyond index 2 are
        # all landmark components, two per landmark.
        landmark_poses = self.state_mean[3:]
        return int(len(landmark_poses) / 2)

    @property
    def landmarks(self):
        # Returns all known landmark positions stacked as an array, in the
        # order they were first observed.
        landmark_array = [self.landmark(label) for label in self._landmark_index.keys()]
        if len(landmark_array) == 0:
            return np.array([])
        return np.stack(landmark_array)

    def landmark(self, label):
        return self.extract_landmark_from_state(label, self.state_mean)

    def pose_as_se2(self) -> se2:
        # Package the pose as an SE(2) rigid transform for downstream geometric
        # operations (composition, inversion) that are awkward with raw vectors.
        x, y, theta = self.pose.flatten()
        return se2(x, y, theta)

    def extract_landmark_from_state(self, label, state_mean):
        # Given a landmark ID, look up its two state rows and return [x, y].
        # Returns None if the filter has never seen this ID.
        if label not in self._landmark_index:
            return None

        state_index = self._landmark_index[label]
        return np.array([state_mean[state_index][0], state_mean[state_index + 1][0]])

    @property
    def state_covariance(self):
        # The full Σ matrix, including all pose-pose, pose-landmark, and
        # landmark-landmark blocks. This is the object that encodes every
        # correlation the filter has learned.
        return np.array(self._state_covariance, copy=True)

    # ------------------------------------------------------------------
    # PREDICT — apply motion model, grow covariance
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
        predicted_robot_pose, F, W = utils.motion_model(pose, motion_command)

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
    # UPDATE — assimilate measurements, register new landmarks
    # ------------------------------------------------------------------

    def update(self, landmark_measurements: List[LandmarkMeasurement]):
        """
        Kalman update step, extended for EKF-SLAM.

        Does two things in order:
          1. Augments the state for any landmark IDs not seen before.
          2. Runs a standard batched Kalman update on all observed landmarks,
             correcting pose and every landmark jointly.

        Math (batched):
            h(μ)   ← stacked expected measurements from all observed landmarks
            C      ← stacked Jacobians (each row: Hr in pose cols, Hl in that lmk's cols)
            y = z - h(μ)                  ← innovation
            S = C Σ Cᵀ + R                ← innovation covariance
            K = Σ Cᵀ S⁻¹                  ← Kalman gain (N × 2m)
            μ⁺ = μ + K y                  ← one correction, every state shifts
            Σ⁺ = (I - K C) Σ              ← every ellipse shrinks
        """
        # No-op if nothing was observed this step — nothing to assimilate.
        if len(landmark_measurements) == 0:
            return

        # Snapshot the prior (μ̄, Σ̄) from the predict step. We'll mutate
        # local copies during state augmentation, then write the final
        # posterior back at the end.
        pose = self.pose
        state_covariance = self.state_covariance
        x = self.state_mean  # Prior state mean μ̄ (will be extended if new lmks appear)

        # Accumulators for the batched update below. Each observed landmark
        # contributes one entry to each list; we stack them later to form
        # the big C, Z, h(μ), and R matrices for a single Kalman update.
        C_list = []
        Z_list = []
        expected_measurements_list = []
        R_list = []

        # ----- STAGE 1: register any new landmarks -----
        # Walk every measurement; if its label is unknown, augment the state.
        # Must happen before the Kalman update so that new landmarks exist in
        # the state vector when we compute their expected measurements below.
        for landmark_measurement in landmark_measurements:
            label = landmark_measurement.label

            if label not in self._landmark_index:
                print(f"Gotten new landmark {landmark_measurement.label}")

                # Inverse sensor model: given the current pose and the relative
                # measurement z, compute where the landmark is in world frame.
                #   l = p + R(θ) · z
                # Also returns:
                #   H1 = G1 = ∂g/∂x  (2×3) — how l shifts with pose error
                #   H2 = G2 = ∂g/∂z  (2×2) — how l shifts with measurement noise
                # Both are needed to give the new landmark a correct covariance
                # (otherwise we'd have to assume it's either perfectly known or
                # infinitely uncertain, both wrong).
                landmark_measured_abs, H1, H2 = utils.inverse_sensor_model(
                    pose, [landmark_measurement.x, landmark_measurement.y]
                )

                # Record where in the state vector this landmark's rows will live.
                # After augmentation the landmark's [x, y] occupies rows
                # [index, index+1].
                index = x.shape[0]
                print(f"Insertion index {index}")
                self._landmark_index[label] = index

                # Append the landmark's absolute position to the state mean.
                x = np.vstack((x, landmark_measured_abs))

                # Build the new landmark's covariance blocks.
                Prr = self.pose_covariance

                # Cross-correlation with the rest of the state (Σ_l,all).
                # The new landmark is a function only of the pose, so its
                # covariance with any existing state block reduces to H1
                # times that pose-related row.
                if len(self._landmark_index) == 1:
                    # First landmark ever: no other landmarks exist, so the
                    # cross block is just H1 · Σ_xx.
                    Plx = np.dot(H1, Prr)
                else:
                    # Subsequent landmarks: the new landmark also co-varies
                    # with every existing landmark (because both depend on
                    # the shared pose). That's captured by stacking Σ_xx and
                    # Σ_xl (the pose-to-existing-landmarks block) and then
                    # left-multiplying by H1.
                    last_state_covariance = deepcopy(state_covariance)
                    Prm = last_state_covariance[0:3, 3:]       # pose-to-known-landmarks
                    Plx = np.dot(H1, np.bmat([[Prr, Prm]]))    # new-lmk-to-[pose, old lmks]

                # Self-covariance of the new landmark.
                #   H1 Σ_xx H1ᵀ  — pose uncertainty propagated into landmark space
                #   H2 R_meas H2ᵀ — sensor noise propagated into landmark space
                # Same additive structure as F Σ Fᵀ + W M Wᵀ in predict:
                # "propagate each uncertain input through its Jacobian and sum".
                Pll = (np.dot(H1, np.dot(Prr, H1.T))
                       + np.dot(H2, np.dot(landmark_measurement.covariance, H2.T)))

                # Extend Σ by one block-row and one block-column:
                #   new Σ = [[  Σ_old   | Plxᵀ ],
                #            [  Plx    |  Pll ]]
                # np.bmat returns a matrix (not ndarray); wrapping in np.array
                # avoids that object leaking out. (np.block is the modern API —
                # swapping to it would be a small polish.)
                P = np.bmat([[state_covariance, Plx.T], [Plx, Pll]])
                state_covariance = np.array(P, copy=True)

        # ----- STAGE 2: build the batched Kalman update -----
        # Now every observed landmark exists in the state. Loop again,
        # building stacked innovation data for one big update.
        for landmark_measurement in landmark_measurements:
            label = landmark_measurement.label

            # Pull the current mean estimate of this landmark out of the
            # (possibly freshly augmented) state vector x.
            estimated_landmark = self.extract_landmark_from_state(label, x)

            # Sensor model h(pose, landmark) — "if the world looked like my
            # current belief, what would the sensor read?"
            # Returns:
            #   expected_measurement — h(μ), the predicted reading
            #   Hr = ∂h/∂x_pose  (2×3) — how reading shifts with pose error
            #   Hl = ∂h/∂l       (2×2) — how reading shifts with this lmk's error
            expected_measurement, Hr, Hl = utils.sensor_model(pose, estimated_landmark)

            # Stack measurement z, expected measurement h(μ), and noise R
            # into the batched matrices.
            Z_list.append(np.array([[landmark_measurement.x], [landmark_measurement.y]]))
            expected_measurements_list.append(expected_measurement)
            R_list.append(landmark_measurement.covariance)

            # Construct this measurement's row of the full C matrix.
            # C is a 2×N matrix where N = 3 + 2·(num landmarks). The measurement
            # depends on ONLY two things in the state: the pose and this specific
            # landmark. So almost every column is zero; just two blocks are nonzero:
            #   columns [0:3]           → Hr (pose derivative)
            #   columns [index:index+2] → Hl (this landmark's derivative)
            # That sparsity is what keeps EKF-SLAM tractable.
            index = self._landmark_index[label]
            C_cols = state_covariance.shape[0]
            C = np.zeros((2, C_cols))
            C[:, :3] = Hr
            C[:, index:index + 2] = Hl
            C_list.append(C)

        # Stack everything vertically for a single batched update.
        # If m landmarks were observed, C becomes (2m × N), Z becomes (2m × 1),
        # R becomes block-diagonal (2m × 2m).
        C = np.vstack(C_list)
        Z = np.vstack(Z_list)
        R = ExtendedKalmanFilter.block_diag(R_list)
        expected_measurements = np.vstack(expected_measurements_list)

        # Innovation: how much the actual measurements disagreed with what the
        # filter predicted. Each pair of rows corresponds to one landmark's
        # (Δx, Δy) mismatch. Nonzero innovation = something to correct.
        y = Z - expected_measurements

        # Innovation covariance: how much surprise should we *expect* here?
        #   C Σ Cᵀ  — state uncertainty projected into measurement space
        #   + R     — add sensor noise
        # Large S means "innovation this big is plausible" → gain will be small.
        # Small S means "innovation this big is shocking" → gain will be big.
        S = C @ state_covariance @ C.T + R

        # Guard against near-singular S (happens when a very confident state
        # meets a very confident measurement in a degenerate configuration).
        # Adding λI regularises without meaningfully changing the physics;
        # strictly a safety net, not part of the math.
        if np.linalg.det(S) < 1e-6:
            S = ExtendedKalmanFilter.reguarlise_matrix(S)

        # Kalman gain. Structurally: (state ↔ measurement cross-covariance)
        # times (innovation covariance)⁻¹. The cross-covariance Σ Cᵀ has
        # entries for EVERY state component, including landmarks far from
        # the observed one — that's how one measurement ends up shifting
        # the entire map through the correlations.
        K = state_covariance @ C.T @ np.linalg.inv(S)

        # Mean correction. K y is an N-vector: pose-x, pose-y, pose-θ,
        # plus corrections for every landmark. A single observation produces
        # a global correction, which is the defining behaviour of joint-state
        # SLAM.
        posterior_state_mean = x + K @ y

        # Covariance contraction. (I − K C) always shrinks Σ in the PSD sense
        # when K carries information, so every diagonal element (every
        # individual variance) decreases. Off-diagonals update too — the
        # filter now knows more about the correlation structure as well.
        I = np.eye(len(posterior_state_mean))
        posterior_state_covariance = (I - K @ C) @ state_covariance

        # Commit the posterior to the filter's internal state. Copying rather
        # than assigning by reference prevents later mutation of the locals
        # from leaking into the filter.
        self._state_vector = np.array(posterior_state_mean, copy=True)
        self._state_covariance = np.array(posterior_state_covariance, copy=True)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def reguarlise_matrix(S, l=0.1):
        # Tikhonov / L2 regularisation: S_reg = S + λI. Used as a fallback
        # when S is near-singular so np.linalg.inv doesn't blow up. The
        # resulting gain is slightly biased toward ignoring the measurement,
        # which is the safe failure mode. Only triggers on pathological cases.
        assert S.shape[0] == S.shape[1], "Matrix S must be square."
        I = np.eye(S.shape[0])
        S_reg = S + l * I
        if np.linalg.det(S_reg) == 0:
            raise ValueError("The regularized matrix is still singular.")
        return S_reg

    @staticmethod
    def block_diag(R_list):
        """
        Build a block-diagonal stacking of per-landmark measurement noise
        covariances. Used because different landmarks (or different viewing
        conditions) can have different R's; stacking them diagonally makes
        the batched update treat each measurement as conditionally independent.

            R = diag(R_1, R_2, ..., R_m)
        """
        rows = sum(R.shape[0] for R in R_list)
        cols = sum(R.shape[1] for R in R_list)
        R = np.zeros((rows, cols))

        row_start, col_start = 0, 0
        for R_i in R_list:
            r, c = R_i.shape
            R[row_start:row_start + r, col_start:col_start + c] = R_i
            row_start += r
            col_start += c

        return R