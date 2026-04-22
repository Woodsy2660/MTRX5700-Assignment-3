"""
Main execution script for EKF SLAM. Coordinates the simulation, EKF, and visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from ekf import ExtendedKalmanFilter
from simulation import Simulation
from math_utils import se2
from evaluation import calculateMapError
from visualization import visualize_slam_animation, visualize_errors

simulation = Simulation(
    grid_size=20,
    num_landmarks=15,
    measurement_noise_std=0.01,       #(meters, isometric)
    linear_odometry_noise_std=0.02,   #(meters, isometric)
    angular_odometry_noise_std=0.01,  #(radians)
    fov_range=10,
    fov_angle=np.pi/2,
    landmarks_file="lmks.txt",
    save_file=True)

ekf = ExtendedKalmanFilter()
ekf_predict = ExtendedKalmanFilter()
trajectory = []
visible_landmarks_history = []
ekf_trajectory_history = []
ekf_predict_trajectory_history = []
ekf_covariance_history = []
ekf_landmarks_history = []

ate_errors = []

num_steps = 800
ekf_update_interval = 2

for t in range(num_steps):
    control_measurement = simulation.step(t)
    true_pose = simulation.get_true_robot_pose()
    trajectory.append(true_pose)

    ekf.predict(control_measurement)
    ekf_predict.predict(control_measurement)

    measurements, visible = simulation.get_landmark_measurements(true_pose)
    visible_landmarks_history.append(visible)

    if t % ekf_update_interval == 0:
        ekf.update(measurements)
    ekf_landmarks_history.append(ekf.landmarks)

    ekf_trajectory_history.append(ekf.pose)
    ekf_covariance_history.append(ekf.pose_covariance)

    ekf_predict_trajectory_history.append(ekf_predict.pose)

    #calculate metric
    true_pose_se2 = se2(data=true_pose)
    estimated_pose_se2 = ekf.pose_as_se2()

    ate_error_se2 = true_pose_se2.inverse() * estimated_pose_se2
    #currently just x,y error
    ate_error = np.linalg.norm(ate_error_se2.matrix()[:2, 2])

    ate_errors.append(ate_error)


# Create two subplots: one for SLAM visualization, one for Error
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 18))

# convert estimate landmarks to dictionary form
landmarks = ekf.landmarks

landmarks_dict = {}
for label in ekf._landmark_index.keys():
  landmarks_dict[label] = ekf.landmark(label)

error_metric = calculateMapError(simulation, landmarks_dict)
print(f"Final Map error is {error_metric}")

# Visualize SLAM with trajectories and covariance
visualize_slam_animation(fig, ax1, simulation, as_animation=False,
                                trajectory=trajectory,
                                visible_landmarks_history=visible_landmarks_history,
                                ekf_trajectory_history=ekf_trajectory_history,
                                ekf_covariance_history=ekf_covariance_history,
                                ekf_landmarks_history= ekf_landmarks_history,
                                ekf_predict_trajectory_history=ekf_predict_trajectory_history
                         )

# Visualize ATE error over time
visualize_errors(fig, ax2, ate_errors)

plt.tight_layout()
plt.show()