import numpy as np

class NaiveGVRController():
    def __init__(self, linear_x, angular_z):
        self.linear_x = linear_x
        self.angular_z = angular_z
    
    def get_action(self):
        # return a naive constant control to constantly going straight, with 0 flipper pos
        return np.array([self.linear_x, self.angular_z, 0.0])