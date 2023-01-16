import numpy as np

class NaiveGVRController():
    def __init__(self, vel):
        self.vel = vel
    
    def get_action(self):
        # return a naive constant control, with 0.0 flipper increment, and constant vel for both wheels
        return np.array([0.0, 0.0, self.vel, self.vel])