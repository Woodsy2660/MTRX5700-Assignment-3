"""
Math utilities, including angle normalization, motion/sensor models, and SE(2) transformations.
Maths models used to reflect the real world into code
"""
import numpy as np

def pi2pi(angle):
    """
    Maps angle to the range of [-pi, pi]
    :param angle: then angle that needs to be mapped to the range [-pi, pi]
    :return : angle in the range [-pi, pi]
    """
    dp = 2*np.pi
    if angle <= -dp or angle >= dp:
        angle = angle % dp
    if angle >= np.pi:
        angle = angle - dp
    if angle <= -np.pi:
        angle = angle + dp
    return angle

class utils:
    @staticmethod
    def motion_model(pose, motion_command):
        """
        Implement the non-linear state-transition function x_k = f(x_{k-1}, u_{k-1})
        which returns x_k and Jacobians H1, H2

        Args:
            pose (_type_): x_{k-1}
            motion_command (_type_):  u_{k-1}

        Returns:
            _type_: new poses and Jacobians df\dx and df\du
        """
        x1, y1, theta1 = pose.flatten()
        dx, dy, dtheta = motion_command.flatten()

        u_rel = motion_command.reshape((3,1))

        R = np.array([[np.cos(theta1), -np.sin(theta1), 0],
            [np.sin(theta1), np.cos(theta1), 0],
            [0, 0, 1]])

        next_robot_pose_abs = R @ u_rel + pose
        # next_robot_pose_abs[2][0] = pi2pi(next_robot_pose_abs[2][0])

        # Calculate Jacobian of X_t+1 with respect to the current robot pose X_t
        F = np.array([[1, 0, -dx*np.sin(theta1)-dy*np.cos(theta1)],
            [0, 1,  dx*np.cos(theta1)-dy*np.sin(theta1)],
            [0, 0, 1]])

        # Calculate Jacobian of X_t+1 with respect to motion command u
        W = np.array([[np.cos(theta1), -np.sin(theta1), 0],
            [np.sin(theta1), np.cos(theta1), 0],
            [0, 0, 1]])

        assert(next_robot_pose_abs.shape == (3,1)), next_robot_pose_abs.shape

        return next_robot_pose_abs, F, W

    @staticmethod
    def inverse_sensor_model(pose, relative_xy):
        """
        Implement the inverse of the sensor model g(x) where, given a relative sensor measurement and pose
        calculate the expected landmark location in the global frame.

        Pose will be an array [x,y,theta] and relative_xy is the [x,y] relative measurements of the landmark.

        Args:
            pose (_type_): np.array (3,1)
            relative_xy (_type_): np.array (2,1)

        Returns:
            _type_: Absolute landmark location and Jacobians df\dx df\dl
        """
        _, _, theta1 = pose.flatten()
        x2, y2 = relative_xy

        landmark_position_rel_vec = np.array([[x2], [y2], [1]])

        # R is the transition matrix to robot frame
        R = np.array([[np.cos(theta1), -np.sin(theta1), 0],
            [np.sin(theta1), np.cos(theta1), 0],
            [0, 0, 1]])

        # Calculate Jacobian G1 with respect to X1
        G1 = np.array([[1, 0, -x2*np.sin(theta1)-y2*np.cos(theta1)],
            [0, 1,  x2*np.cos(theta1)-y2*np.sin(theta1)]])

        # Calculate Jacobian G2 with respect to X2
        G2 = np.array([[np.cos(theta1), -np.sin(theta1)],
            [np.sin(theta1),  np.cos(theta1)]])

        landmark_abs = np.array(np.dot(R, landmark_position_rel_vec)) + np.array(pose)

        return np.array([[landmark_abs[0][0]], [landmark_abs[1][0]]]), G1, G2



        # return absolute_xy, H1, H2

    @staticmethod
    def sensor_model(pose, absolute_xy):
        """
            Implement sensor model h(x) where, given a absolute sensor measurement and pose
            calculate the expected landmark location in the local frame.

            This is used to calculate the expected measurement (in the local frame) given the current estimate
            of the robot and landmark

            Pose will be an array [x,y,theta] and absolute_xy is the [x,y] relative measurements of the landmark.

            Args:
                pose (_type_): np.array (3,1)
                absolute_xy (_type_): np.array (2,1)

            Returns:
                _type_: relative landmark location and Jacobians df\dx df\dl
        """
        x1, y1, theta1 = pose.flatten()
        x2, y2 = absolute_xy

        diff = np.array([[x2-x1],
            [y2-y1],
            [1]])

        # R is the transition matrix to robot frame
        R = [[np.cos(-theta1), -np.sin(-theta1), 0],
            [np.sin(-theta1), np.cos(-theta1), 0],
            [0, 0, 1]]

        landmark_position_rel = np.dot(R, diff)

        # Calculate Jacobian of the relative landmark position wrt. the robot pose,
        # i.e. [x1, y1, theta1]
        H = np.array([[-np.cos(theta1), -np.sin(theta1), -(x2-x1)*np.sin(theta1)+(y2-y1)*np.cos(theta1)],
            [np.sin(theta1), -np.cos(theta1), -(x2-x1)*np.cos(theta1)-(y2-y1)*np.sin(theta1)]])

        # Calculate Jacobian of the relative landmark position wrt. the absolute
        # landmark pose. i.e. [x2, y2]
        J = np.array([[np.cos(theta1), np.sin(theta1)],
            [-np.sin(theta1), np.cos(theta1)]])

        return np.array([[landmark_position_rel[0][0]], [landmark_position_rel[1][0]]]), H, J

class se2:

    def __init__(self, x: float = None, y: float = None, theta: float = None, data=None):
        """Initialize SE(2) transform with either:
        - explicit x, y, theta (radians), or
        - a list/array [x, y, theta].
        """
        if data is not None:
            if isinstance(data, np.ndarray):
                data = list(data.flatten())
            if isinstance(data, list) and len(data) == 3:
                self.x, self.y, self.theta = data
                assert isinstance(self.x, float), f"x is of type {type(self.x)}, {self.x}"
                assert isinstance(self.y, float), f"y is of type {type(self.y)}"
                assert isinstance(self.theta, float), f"theta is of type {type(self.theta)}"
            else:
                raise ValueError("data must be a list or array of length 3.")
        elif x is not None and y is not None and theta is not None:
            self.x = x
            self.y = y
            self.theta = theta
        else:
            raise ValueError("Provide either (x, y, theta) or a list/array of size 3.")
        # store cos(theta)
        self._c = np.cos(self.theta)
        # store sin(theta)
        self._s = np.sin(self.theta)

    def matrix(self):
        return np.array([
            [self._c, -self._s, self.x],
            [self._s,  self._c, self.y],
            [0,  0, 1]
        ])

    def compose(self, other):
        """Compose this transform with another SE(2) transform."""
        result_matrix = self.matrix() @ other.matrix()
        x, y = result_matrix[:2, 2]
        theta = np.arctan2(result_matrix[1, 0], result_matrix[0, 0])
        return se2(x, y, theta)

    def inverse(self):
        """Compute the inverse of this transform."""
        inv_x = -self._c * self.x - self._s * self.y
        inv_y =  self._s * self.x - self._c * self.y
        return se2(inv_x, inv_y, -self.theta)

    def __mul__(self, other: "se2") -> "se2":
        """Overload the multiplication operator for composition."""
        return self.compose(other)

