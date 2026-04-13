"""
Created on Tue Aug 9 16:12:08 2016
Calculate the robot pose given the robot previous pose and motion of the robot
with respect to that pose
@author: admin-u5941570
Updated on 26-3-23
@maintainers: Viorela Ila, Max Revay, Jing Cheng, Stefan Williams,
Stephany Berrio Perez, Tejaswi Digumarti, Jesse Morris, Arihant Lunawat
"""

import numpy as np


def pi2pi(angle):
    """
    Maps angle to the range of [-pi, pi]
    :param angle: then angle that needs to be mapped to the range [-pi, pi]
    :return : angle in the range [-pi, pi]
    """

    dp = 2 * np.pi
    if angle <= -dp or angle >= dp:
        angle = angle % dp
    if angle >= np.pi:
        angle = angle - dp
    if angle <= -np.pi:
        angle = angle + dp

    return angle


def Relative2AbsolutePose(robot_pose_abs, u_rel):
    """
    Calculates the new pose of the robot given its current pose in the
    absolute coordinate frame and a motion input.

    :param robot_pose_abs: current pose of the robot in the absolute reference
    frame [x, y, theta]
    :type robot_pose_abs: [type]
    :param u_rel: motion command in the robot's frame of reference
    [dx, dy, dtheta]
    :type u_rel: [type]
    :return: pose of the robot in the absolute reference frame after applying
    the motion and the Jacobians of the new pose wrt, the current pose
    and the motion command respectively
    :rtype: tuple
    """

    assert robot_pose_abs.shape == (3, 1)
    assert u_rel.shape == (3, 1)

    x1 = robot_pose_abs[0][0]
    y1 = robot_pose_abs[1][0]
    theta1 = robot_pose_abs[2][0]
    dx = u_rel[0][0]
    dy = u_rel[1][0]
    dtheta = u_rel[2][0]

    # R is the transition matrix of robot frame
    # i.e. X_t+1 = X_t + R(theta_t) * u
    R = np.array(
        [
            [np.cos(theta1), -np.sin(theta1), 0],
            [np.sin(theta1), np.cos(theta1), 0],
            [0, 0, 1],
        ]
    )

    next_robot_pose_abs = R @ u_rel + robot_pose_abs
    next_robot_pose_abs[2][0] = pi2pi(next_robot_pose_abs[2][0])

    # Calculate Jacobian of X_t+1 with respect to the current robot pose X_t
    F = np.array(
        [
            [1, 0, -dx * np.sin(theta1) - dy * np.cos(theta1)],
            [0, 1, dx * np.cos(theta1) - dy * np.sin(theta1)],
            [0, 0, 1],
        ]
    )

    # Calculate Jacobian of X_t+1 with respect to motion command u
    W = np.array(
        [
            [np.cos(theta1), -np.sin(theta1), 0],
            [np.sin(theta1), np.cos(theta1), 0],
            [0, 0, 1],
        ]
    )

    assert next_robot_pose_abs.shape == (3, 1)

    return next_robot_pose_abs, F, W


def Absolute2RelativeXY(robot_pose_abs, landmark_position_abs):
    """Converts a landmark's position from the absolute frame of reference to
    the robot's frame of reference, i.e the position of the landmarks as
    measured by the robot.

    :param robot_pose_abs: pose of the robot in the absolute frame of
    reference [x, y, theta]. 3x1 array
    :type robot_pose_abs: np.array
    :param landmark_position_abs: position of the landmark in the absolute
    frame of reference [x, y]. 2x1 array
    :type landmark_position_abs: np.array
    :return: position of the landmark in the to the robot's frame of reference
    [x, y], and the Jacobians of the measurement model with respect to the robot
    pose and the landmark
    :rtype: tuple
    """

    assert robot_pose_abs.shape == (3, 1)
    assert landmark_position_abs.shape == (2, 1)

    x1 = robot_pose_abs[0][0]
    y1 = robot_pose_abs[1][0]
    theta1 = robot_pose_abs[2][0]
    x2 = landmark_position_abs[0]
    y2 = landmark_position_abs[1]

    # Calculate the difference with respect to world frame
    diff = np.array([[x2 - x1], [y2 - y1], [1]])

    # R is the transition matrix to robot frame
    R = [
        [np.cos(-theta1), -np.sin(-theta1), 0],
        [np.sin(-theta1), np.cos(-theta1), 0],
        [0, 0, 1],
    ]

    landmark_position_rel = np.dot(R, diff)

    # Calculate Jacobian of the relative landmark position wrt. the robot pose,
    # i.e. [x1, y1, theta1]
    H = np.array(
        [
            [
                -np.cos(theta1),
                -np.sin(theta1),
                -(x2 - x1) * np.sin(theta1) + (y2 - y1) * np.cos(theta1),
            ],
            [
                np.sin(theta1),
                -np.cos(theta1),
                -(x2 - x1) * np.cos(theta1) - (y2 - y1) * np.sin(theta1),
            ],
        ]
    )

    # Calculate Jacobian of the relative landmark position wrt. the absolute
    # landmark pose. i.e. [x2, y2]
    J = np.array([[np.cos(theta1), np.sin(theta1)], [-np.sin(theta1), np.cos(theta1)]])

    return (
        np.array([[landmark_position_rel[0][0]], [landmark_position_rel[1][0]]]),
        H,
        J,
    )


def Relative2AbsoluteXY(robot_pose_abs, landmark_position_rel):
    """
    Convert's a landmark's position from the robot's frame of reference to the absolute frame of reference
    :param robot_pose_abs: pose of the robot in the the absolute frame of reference [x, y, theta]
    :param landmark_position_rel: position of the landmark in the robot's frame of reference [x, y]
    :return : [position of the landmark in the absolute frame of reference [x, y], G1, G2]
    """

    assert robot_pose_abs.shape == (3, 1)
    assert landmark_position_rel.shape == (2, 1)

    x1 = robot_pose_abs[0][0]
    y1 = robot_pose_abs[1][0]
    theta1 = robot_pose_abs[2][0]
    x2 = landmark_position_rel[0]
    y2 = landmark_position_rel[1]

    landmark_position_rel_vec = np.array([[x2], [y2], [1]])

    # R is the transition matrix to robot frame
    R = np.array(
        [
            [np.cos(theta1), -np.sin(theta1), 0],
            [np.sin(theta1), np.cos(theta1), 0],
            [0, 0, 1],
        ]
    )

    # Calculate Jacobian H1 with respect to X1
    G1 = np.array(
        [
            [1, 0, -x2 * np.sin(theta1) - y2 * np.cos(theta1)],
            [0, 1, x2 * np.cos(theta1) - y2 * np.sin(theta1)],
        ]
    )

    # Calculate Jacobian H2 with respect to X2
    G2 = np.array([[np.cos(theta1), -np.sin(theta1)], [np.sin(theta1), np.cos(theta1)]])

    landmark_abs = np.array(np.dot(R, landmark_position_rel_vec)) + np.array(
        robot_pose_abs
    )

    return np.array([[landmark_abs[0][0]], [landmark_abs[1][0]]]), G1, G2


def RelativeLandmarkPositions(landmark_position_abs, next_landmark_position_abs):
    """
    Given two input landmark positions in the absolute frame of reference, computes the relative position of the
    next landmark with respect to the current landmark
    :param landmark_position_abs: position of the current landmark in the absolute reference frame [x, y]
    :param next_landmark_position_abs: position of the next landmark in the absolute reference frame [x, y]
    :return : relative position of the next landmark with respect to the current landmark's position [dx, dy]
    """

    assert landmark_position_abs.shape == (2, 1)
    assert next_landmark_position_abs.shape == (2, 1)

    # This function does not need any changes

    # label is in position [0], hence use positions [1] and [2]
    x1 = float(landmark_position_abs[1])
    y1 = float(landmark_position_abs[2])
    x2 = float(next_landmark_position_abs[1])
    y2 = float(next_landmark_position_abs[2])

    # Calculate the difference of position in world frame
    diff = [x2 - x1, y2 - y1]

    return diff


def homogenous_transform(R: np.array, t: np.array):
    """
    Given a rotation matrix R and a translation vector t, compute the homogenous transformation matrix H
    that transforms points from the local frame to the global frame.
    """

    assert t.shape == (3, 1)
    assert R.shape == (3, 3)

    # This function does not need any changes

    H = np.eye(4)
    H[:3, :3] = R
    t = np.transpose(t)
    H[:3, 3] = t[:3]

    return H
