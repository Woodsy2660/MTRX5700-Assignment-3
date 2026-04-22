"""
Data structures for control and landmark measurements.
"""
class LandmarkMeasurement:
    def __init__(self, label, x, y, covariance):
        self.label = label
        self.x = x
        self.y = y
        self.covariance = covariance

class ControlMeasurement:
    def __init__(self, motion_vector, covariance):
        self.motion_vector = motion_vector
        self.covariance = covariance

