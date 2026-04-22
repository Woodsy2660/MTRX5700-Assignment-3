"""
Visualization utilities for SLAM, including animation and error plotting.
"""
import numpy as np

from matplotlib import patches
import matplotlib.animation as animation

def plot_covariance_ellipse_2d(axes, origin, covariance: np.ndarray, ellipse_patch=None):
    """
    Plots a Gaussian as an uncertainty ellipse scaled for 95% inliers.
    """
    w, v = np.linalg.eigh(covariance)

    # Fix: Handle numerical issues where eigenvalues might be slightly negative
    w = np.maximum(w, 0)

    k = 2.447746830681
    x,y,_ = origin

    angle = np.arctan2(v[1, 0], v[0, 0])

    if ellipse_patch is None:
        e1 = patches.Ellipse((x, y),
                             np.sqrt(w[0]) * 2 * k,
                             np.sqrt(w[1]) * 2 * k,
                             angle=np.rad2deg(angle),
                             fill=False,
                             color='orange',
                             label='Covariance')
        axes.add_patch(e1)
        return e1
    else:
        ellipse_patch.center = (x, y)
        ellipse_patch.width = np.sqrt(w[0]) * 2 * k
        ellipse_patch.height = np.sqrt(w[1]) * 2 * k
        ellipse_patch.angle = np.rad2deg(angle)
        return ellipse_patch

def visualize_slam_animation(fig, ax, simulation, as_animation=False, **kwargs):
    trajectory = kwargs.get("trajectory", None)
    visible_landmarks_history = kwargs.get("visible_landmarks_history", None)
    ekf_trajectory_history = kwargs.get("ekf_trajectory_history", None)
    ekf_covariance_history = kwargs.get("ekf_covariance_history", None)
    ekf_landmarks_history = kwargs.get("ekf_landmarks_history", None)
    ekf_predict_trajectory_history = kwargs.get("ekf_predict_trajectory_history", None)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('SLAM Simulation Visualization')
    ax.grid(True)
    ax.set_aspect('equal')

    x_landmarks = [lm[1] for lm in simulation.landmarks]
    y_landmarks = [lm[2] for lm in simulation.landmarks]
    ax.scatter(x_landmarks, y_landmarks, marker='x', color='blue', label='Ground Truth Landmarks')

    axis_elements = []
    trajectory_plot, = ax.plot([], [], marker='.', linestyle='-', color='green', label='True Robot Trajectory')
    axis_elements.append(trajectory_plot)

    ekf_trajectory_plot, = ax.plot([], [], marker='.', linestyle='-', color='blue', label='EKF Robot Trajectory')
    axis_elements.append(ekf_trajectory_plot)

    if ekf_predict_trajectory_history:
        ekf_predict_trajectory_plot, = ax.plot([], [], marker='.', linestyle='-', color='red', label='EKF (Predict Only)')
        axis_elements.append(ekf_predict_trajectory_plot)

    visible_landmarks_plot = ax.scatter([], [], marker='o', color='red', label='Visible Landmarks')
    axis_elements.append(visible_landmarks_plot)

    estimated_landmarks_plot = ax.scatter([], [], marker='x', color='green', label='Estimated Landmarks')
    axis_elements.append(estimated_landmarks_plot)

    ellipse_patch = None

    def update(frame):
        nonlocal ellipse_patch
        current_traj = np.array(trajectory[:frame + 1])
        trajectory_plot.set_data(current_traj[:, 0, 0], current_traj[:, 1, 0])

        current_ekf_traj = np.array(ekf_trajectory_history[:frame + 1])
        ekf_trajectory_plot.set_data(current_ekf_traj[:, 0], current_ekf_traj[:, 1])

        if ekf_predict_trajectory_history:
            current_pred_traj = np.array(ekf_predict_trajectory_history[:frame + 1])
            ekf_predict_trajectory_plot.set_data(current_pred_traj[:, 0], current_pred_traj[:, 1])

        if ekf_landmarks_history and frame < len(ekf_landmarks_history) and len(ekf_landmarks_history[frame]) > 0:
            est_lmks = ekf_landmarks_history[frame]
            estimated_landmarks_plot.set_offsets(est_lmks)

        if visible_landmarks_history and frame < len(visible_landmarks_history):
            vis_x = [simulation.landmarks[i][1] for i in visible_landmarks_history[frame]]
            vis_y = [simulation.landmarks[i][2] for i in visible_landmarks_history[frame]]
            visible_landmarks_plot.set_offsets(np.column_stack((vis_x, vis_y)))

        if ekf_covariance_history and frame < len(ekf_covariance_history):
            mu = ekf_trajectory_history[frame]
            P = ekf_covariance_history[frame][:2, :2]
            ellipse_patch = plot_covariance_ellipse_2d(ax, mu, P, ellipse_patch)

        return *axis_elements, ellipse_patch

    if as_animation:
        return animation.FuncAnimation(fig, update, frames=len(trajectory), interval=20, blit=True)
    else:
        for f in range(len(trajectory)): update(f)
        ax.legend()
        return fig

def visualize_errors(fig, ax, error_metric):
    ax.set_xlabel('Step')
    ax.set_ylabel('Translation Error (m)')
    ax.set_title('Absolute Trajectory Error over Time')
    ax.grid(True)
    ax.plot(error_metric, color='purple', label='ATE')
    ax.legend()

