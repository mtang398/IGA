import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from matplotlib import cm
import time
from scipy.linalg import solve
from scipy.special import comb
from copy import deepcopy

import numpy as np
import sympy as sp
from geomdl import NURBS
from geomdl.visualization import VisMPL
from geomdl import operations

# Define symbolic sqrt function
sqrt = sp.sqrt

# Control points coordinates and weights
cp_data = np.array([
    [[4*(1 - sqrt(3)), 4*(1 - sqrt(3)), 4*(1 - sqrt(3)), 4*(3 - sqrt(3))],
     [-sqrt(2), sqrt(2)*(sqrt(3) - 4), sqrt(2)*(sqrt(3) - 4), sqrt(2)*(3*sqrt(3) - 2)],
     [0, 4*(1 - 2*sqrt(3))/3, 4*(1 - 2*sqrt(3))/3, 4*(5 - sqrt(3))/3],
     [sqrt(2), sqrt(2)*(sqrt(3) - 4), sqrt(2)*(sqrt(3) - 4), sqrt(2)*(3*sqrt(3) - 2)],
     [4*(sqrt(3) - 1), 4*(1 - sqrt(3)), 4*(1 - sqrt(3)), 4*(3 - sqrt(3))]],
    
    [[sqrt(2)*(sqrt(3) - 4), -sqrt(2), sqrt(2)*(sqrt(3) - 4), sqrt(2)*(3*sqrt(3) - 2)],
     [(2 - 3*sqrt(3))/2, (2 - 3*sqrt(3))/2, -(sqrt(3) + 6)/2, (sqrt(3) + 6)/2],
     [0, sqrt(2)*(2*sqrt(3) - 7)/3, -5*sqrt(6)/3, sqrt(2)*(sqrt(3) + 6)/3],
     [(3*sqrt(3) - 2)/2, (2 - 3*sqrt(3))/2, -(sqrt(3) + 6)/2, (sqrt(3) + 6)/2],
     [sqrt(2)*(4 - sqrt(3)), -sqrt(2), sqrt(2)*(sqrt(3) - 4), sqrt(2)*(3*sqrt(3) - 2)]],
    
    [[4*(1 - 2*sqrt(3))/3, 0, 4*(1 - 2*sqrt(3))/3, 4*(5 - sqrt(3))/3],
     [sqrt(2)*(2*sqrt(3) - 7)/3, 0, -5*sqrt(6)/3, sqrt(2)*(sqrt(3) + 6)/3],
     [0, 0, 4*(sqrt(3) - 5)/3, 4*(5*sqrt(3) - 1)/9],
     [sqrt(2)*(7 - 2*sqrt(3))/3, 0, -5*sqrt(6)/3, sqrt(2)*(sqrt(3) + 6)/3],
     [4*(2*sqrt(3) - 1)/3, 0, 4*(1 - 2*sqrt(3))/3, 4*(5 - sqrt(3))/3]],
    
    [[sqrt(2)*(sqrt(3) - 4), sqrt(2), sqrt(2)*(sqrt(3) - 4), sqrt(2)*(3*sqrt(3) - 2)],
     [(2 - 3*sqrt(3))/2, (3*sqrt(3) - 2)/2, -(sqrt(3) + 6)/2, (sqrt(3) + 6)/2],
     [0, sqrt(2)*(7 - 2*sqrt(3))/3, -5*sqrt(6)/3, sqrt(2)*(sqrt(3) + 6)/3],
     [(3*sqrt(3) - 2)/2, (3*sqrt(3) - 2)/2, -(sqrt(3) + 6)/2, (sqrt(3) + 6)/2],
     [sqrt(2)*(4 - sqrt(3)), sqrt(2), sqrt(2)*(sqrt(3) - 4), sqrt(2)*(3*sqrt(3) - 2)]],
    
    [[4*(1 - sqrt(3)), 4*(sqrt(3) - 1), 4*(1 - sqrt(3)), 4*(3 - sqrt(3))],
     [-sqrt(2), sqrt(2)*(4 - sqrt(3)), sqrt(2)*(sqrt(3) - 4), sqrt(2)*(3*sqrt(3) - 2)],
     [0, 4*(2*sqrt(3) - 1)/3, 4*(1 - 2*sqrt(3))/3, 4*(5 - sqrt(3))/3],
     [sqrt(2), sqrt(2)*(4 - sqrt(3)), sqrt(2)*(sqrt(3) - 4), sqrt(2)*(3*sqrt(3) - 2)],
     [4*(sqrt(3) - 1), 4*(sqrt(3) - 1), 4*(1 - sqrt(3)), 4*(3 - sqrt(3))]]
], dtype=object)

# Convert control_points and weights to float data type
control_points_bottom_float = cp_data[..., :3].astype(float)
weights_bottom_float = cp_data[..., 3].astype(float)

knot_vector_u_bottom = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
knot_vector_v_bottom = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

def compute_actual_control_points(weighted_control_points, weights, radius):
    return weighted_control_points / weights[..., None] * radius

actual_control_points_bottom = compute_actual_control_points(control_points_bottom_float, weights_bottom_float, 1)