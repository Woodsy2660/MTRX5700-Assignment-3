"""
Evaluation metrics for the SLAM implementation, such as Map Error.
"""
import numpy as np
from simulation import Simulation

def calculateMapError(simulation: Simulation, landmarks_ekf):
    """
    Calculates the error between ground truth data and the data obtained after SLAM and returns the total error.
    landmarks ekf should be a dictionary mapping the label id (index) of the landmark to its position in the global frame
    e.g. {1: [x, y], 2: [x,y]}...
    """

    # N x 3 array in the form (label, x, y)
    landmarks_ground_truth = simulation.get_landmarks()

    if len(landmarks_ekf) == 0:
      print("No landmarks found ")

    landmark_gt = {}

    # convert gt landmarks to map
    for landmark in landmarks_ground_truth:
        landmark_gt[landmark[0]] = [landmark[1], landmark[2]]

    # Compute positions of landmarks relative to the positions of another landmark
    landmark_error = []
    landmark_gt_ids = landmark_gt.keys()


    missed_lmks = []
    for landmark_id, _ in landmark_gt.items():
        if landmark_id not in landmarks_ekf:
            print(f"Ground truth lmk id {landmark_id} does not appear in your estimate!")
            missed_lmks.append(landmark_id)

    additional_lmks = []
    matched_mks = []

    for landmark_id, relative_pos in landmarks_ekf.items():
        if landmark_id not in landmark_gt_ids:
            additional_lmks.append(landmark_id)
            print(f"Your solution has lmk id {landmark_id} which does not occur in the ground truth")
        else:
            matched_mks.append(landmark_id)

    def relativeLandmarkPositions(landmark_position_abs, next_landmark_position_abs):
      # label is in position [0], hence use positions [1] and [2]
      x1 = float(landmark_position_abs[1])
      y1 = float(landmark_position_abs[2])
      x2 = float(next_landmark_position_abs[1])
      y2 = float(next_landmark_position_abs[2])

      # Calculate the difference of position in world frame
      diff = [x2-x1, y2-y1]

      return diff


    for i in range(0, len(matched_mks) - 1):
      # Compute the Absolute Trajectory Error
        lmk_id_1 = matched_mks[i]
        lmk_id_2 = matched_mks[i+1]

        predicted_pos_1 = [lmk_id_1, landmarks_ekf[lmk_id_1][0], landmarks_ekf[lmk_id_1][1]]
        gt_pos_1 = [lmk_id_1, landmark_gt[lmk_id_1][0], landmark_gt[lmk_id_1][1]]

        predicted_pos_2 = [lmk_id_2, landmarks_ekf[lmk_id_2][0], landmarks_ekf[lmk_id_2][1]]
        gt_pos_2 = [lmk_id_2, landmark_gt[lmk_id_2][0], landmark_gt[lmk_id_2][1]]

        landmarks_pred_rel = relativeLandmarkPositions(predicted_pos_1,predicted_pos_2)
        landmarks_gt_rel = relativeLandmarkPositions(gt_pos_1, gt_pos_2)

        landmark_error.append(np.array(landmarks_gt_rel) - np.array(landmarks_pred_rel))

    if len(landmarks_ekf) > 1:
        error_landmark = (1.0/(len(landmarks_ekf)-1))*np.linalg.norm(landmark_error)
    else:
        error_landmark = 0.

    return error_landmark

