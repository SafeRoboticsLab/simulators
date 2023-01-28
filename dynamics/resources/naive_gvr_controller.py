import numpy as np

class NaiveGVRController():
    def __init__(self, vel):
        self.vel = vel
    
    def get_action(self):
        # return a naive constant control to constantly going straight, with 0 flipper pos
        return np.array([self.vel, 0.0, 0.0])