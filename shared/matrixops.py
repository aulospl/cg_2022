import numpy as np
import math

def cos(d):
    rad = (math.pi / 180) * d
    return math.cos(rad)

def sin(d):
    rad = (math.pi / 180) * d
    return math.sin(rad)

def translate_2d(vert, dx, dy):
    tra = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
    return np.mapmul(tra, vert)

def rotate_2d(vert, deg):
    rot = np.array([[cos(deg), -sin(deg), 0], [sin(deg), cos(deg), 0], [0, 0, 1]])
    return np.matmul(rot, vert)

def scale_2d(vert, sx, sy):
    sca = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
    return np.matmul(sca, vert)

def reflection_2d(vert, axis):
    if axis == 'h':
        ref = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        return np.matmul(ref, vert)
    elif axis == 'v':
         ref = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
         return np.matmul(ref, vert)
    elif axis == 'd':
         ref = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
         return np.matmul(ref, vert)
    else:
        return "ERROR: unknown axis "

def shearing_2d(vert, shx):
    shear = np.array([[1, shx, 0], [0, 1, 0], [0, 0, 1]])
    return np.matmul(shear, vert)

