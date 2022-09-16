import numpy as np
import math

def rotate_x(theta):
    return np.array([
        [1, 0, 0],
        [0, math.cos(theta), -math.sin(theta)],
        [0, math.sin(theta), math.cos(theta)]
    ])

def translate_z(d):
    return np.array([
        [0],
        [0],
        [d]
    ])

def rotate_z(theta):
    return np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0],
        [0, 0, 1]
    ])

def translate_x(d):
    return np.array([
        [d],
        [0],
        [0]
    ])

def rotate_y(theta):
    return np.array([
        [math.cos(theta), 0, math.sin(theta)],
        [0, 1, 0],
        [-math.sin(theta), 0, math.cos(theta)]
    ])

def translate_y(d):
    return np.array([
        [0],
        [d],
        [0]
    ])