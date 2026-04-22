"""
Simulation environment for generating robot trajectories and landmark measurements.
"""
import numpy as np
import random
import os
import pickle

from math_utils import utils
from measurements import LandmarkMeasurement, ControlMeasurement

class Simulation:
    def __init__(self, grid_size, num_landmarks, measurement_noise_std, linear_odometry_noise_std, angular_odometry_noise_std, fov_range, fov_angle, **kwargs):
        self.grid_size = grid_size
        self.num_landmarks = num_landmarks
        self.measurement_noise_std = measurement_noise_std
        self.linear_odometry_noise_std = linear_odometry_noise_std
        self.angular_odometry_noise_std = angular_odometry_noise_std
        self.robot_pose = np.array([[0.0], [0.0], [0.0]])  # Initial robot pose
        self.fov_range = fov_range
        self.fov_angle = fov_angle
        self.dt = 0.1 #Time step
        self.target_landmark_index = 0

        lmks_file_path = kwargs.get("landmarks_file", None)
        save_file = kwargs.get("save_file", False)

        if lmks_file_path is not None:
            try:
                self.landmarks = self.load_landmarks(lmks_file_path)
            except Exception as e:
                print(f"Failed to load lmks from file. Geneating new lmks!")
                self.landmarks = self.generate_landmarks()
        else:
            self.landmarks = self.generate_landmarks()

        if save_file:
            self.save_landmarks(file_path="lmks.txt")


    def generate_landmarks(self):
        landmarks = []
        for i in range(self.num_landmarks):
            x = random.uniform(-self.grid_size / 2, self.grid_size / 2)
            y = random.uniform(-self.grid_size / 2, self.grid_size / 2)
            landmarks.append((i, x, y))  # (label, x, y)
        return landmarks

    def load_landmarks(self, file_path):
        print(f"Loading lmks from file {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            if not isinstance(data, list) or not all(isinstance(item, tuple) for item in data):
                raise ValueError("The file does not contain a valid list of tuples.")

            return data
        except (IOError, pickle.UnpicklingError) as e:
            raise IOError(f"Error reading the file {file_path}: {e}")

    def save_landmarks(self, file_path="lmks.txt"):
        print(f"Saving lmks to file {file_path}")
        if not isinstance(self.landmarks, list) or not all(isinstance(item, tuple) for item in self.landmarks):
            raise ValueError("Data must be a list of tuples.")

        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.landmarks, f)
        except IOError as e:
            raise IOError(f"Error writing to file {file_path}: {e}")


    def generate_odometry_measurement(self, delta_d, delta_theta):
        # Add noise to odometry
        noisy_delta_d = delta_d + np.random.normal(0, self.linear_odometry_noise_std)
        noisy_delta_theta = delta_theta + np.random.normal(0, self.angular_odometry_noise_std)

        # no control along y
        dy = 0
        #just a small value as there is no motion along y of the robot
        s_linear_vel_y = 0.000000001

        # Update true robot pose
        self.robot_pose, _, _ = utils.motion_model(self.robot_pose, np.array([[delta_d], [dy], [delta_theta]]))

        # Create ControlMeasurement object
        covariance = np.diag([self.linear_odometry_noise_std**2, s_linear_vel_y**2, self.angular_odometry_noise_std**2])
        return ControlMeasurement(np.array([[noisy_delta_d], [dy], [noisy_delta_theta]]), covariance)

    def generate_landmark_measurement(self, landmark_index, robot_pose):
        label, lx, ly = self.landmarks[landmark_index]

        relative_l, _, _ =utils.sensor_model(robot_pose, np.array([lx, ly]))
        relative_x, relative_y = relative_l.flatten()

        # # Add noise to measurement
        noisy_relative_x = relative_x + np.random.normal(0, self.measurement_noise_std)
        noisy_relative_y = relative_y + np.random.normal(0, self.measurement_noise_std)

        # Create LandmarkMeasurement object
        covariance = np.diag([self.measurement_noise_std**2, self.measurement_noise_std**2])
        return LandmarkMeasurement(label, noisy_relative_x, noisy_relative_y, covariance)

    def get_true_robot_pose(self):
        return self.robot_pose

    def get_landmarks(self):
        return self.landmarks

    def step(self, t):
        # Generate control input based on time
        delta_d, delta_theta = self.generate_control_input(t)

        # Move the robot with noise
        control_measurement = self.generate_odometry_measurement(delta_d, delta_theta)
        return control_measurement


    def get_landmark_measurements(self, pose):
        visible_landmarks = self.get_visible_landmarks(pose)
        measurements = []
        for landmark_index in visible_landmarks:
            measurements.append(self.generate_landmark_measurement(landmark_index, pose))
        return measurements, visible_landmarks


    #     return delta_d, delta_theta
    def generate_control_input(self, t):
      """Generates control inputs to visit landmarks."""
      rx, ry, rtheta = self.robot_pose.flatten()
      target_lx, target_ly = self.landmarks[self.target_landmark_index][1], self.landmarks[self.target_landmark_index][2]

      delta_x = target_lx - rx
      delta_y = target_ly - ry
      angle_to_target = np.arctan2(delta_y, delta_x)

      # Calculate angle difference
      angle_diff = angle_to_target - rtheta
      angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

      # Calculate distance to target
      distance_to_target = np.sqrt(delta_x**2 + delta_y**2)

    # Control parameters
      angular_speed_gain = 0.5  # Proportional gain for angular speed
      linear_speed_gain = 0.1  # Proportional gain for linear speed
      max_angular_speed = 1.0 # Maximum allowed angular speed
      max_linear_speed = 0.5 # Maximum allowed linear speed

      # Generate control inputs with proportional gains
      delta_theta = angular_speed_gain * angle_diff
      delta_d = linear_speed_gain * distance_to_target

      # Clip values to maximum speeds multiplied by dt
      delta_theta = np.clip(delta_theta, -max_angular_speed * self.dt, max_angular_speed * self.dt)
      delta_d = np.clip(delta_d, 0, max_linear_speed * self.dt)


    #   # Add some randomness
    #   delta_d += np.random.normal(0, 0.05)
    #   delta_theta += np.random.normal(0, 0.01)

      # Check if reached the target
      if distance_to_target < 3.0:  # Threshold for reaching the target
          self.target_landmark_index = (self.target_landmark_index + 2) % len(self.landmarks)

      return delta_d, delta_theta

    def get_visible_landmarks(self, robot_pose):
        visible_landmarks = []
        rx, ry, rtheta = robot_pose.flatten()
        for i, (label, lx, ly) in enumerate(self.landmarks):
            delta_x = lx - rx
            delta_y = ly - ry
            range_to_landmark = np.sqrt(delta_x**2 + delta_y**2)
            angle_to_landmark = np.arctan2(delta_y, delta_x) - rtheta

            # Normalize angle to [-pi, pi]
            angle_to_landmark = (angle_to_landmark + np.pi) % (2 * np.pi) - np.pi

            if range_to_landmark <= self.fov_range and abs(angle_to_landmark) <= self.fov_angle / 2:
                visible_landmarks.append(i)
        return visible_landmarks

