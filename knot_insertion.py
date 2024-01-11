import numpy as np

# Monkey-patching np.float
if not hasattr(np, 'float'):
    np.float = float

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def compute_actual_control_points(weighted_control_points, weights, radius):
    return weighted_control_points / weights[..., None] * radius

actual_control_points_bottom = compute_actual_control_points(control_points_bottom_float, weights_bottom_float, 1)

# Combine control_points and weights
control_points_4d = np.concatenate([control_points_bottom_float, weights_bottom_float[..., None]], axis=-1)

# Create a NURBS surface object
surf = NURBS.Surface()

# Setting the degrees and knot vectors
surf.degree_u = 4  # Degree in the u direction
surf.degree_v = 4  # Degree in the v direction

# Set the weighted control points
surf.ctrlpts2d = control_points_4d.tolist()

surf.knotvector_u = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
surf.knotvector_v = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

def perform_insert_knot(surf, h_ref_order):
    if h_ref_order == 0:
        # Do nothing
        return
    
    num_insertions = 4  # number of times the knot to be inserted
    interval = 1 / (2 ** h_ref_order)  # interval directly calculated from h_ref_order
    
    # Insert knots at intervals
    for insertion_point in np.arange(interval, 1, interval):
        operations.insert_knot(surf, [insertion_point, insertion_point], [num_insertions, num_insertions])


perform_insert_knot(surf, 2)

# print(np.array(surf._control_points2D)[...,:3].reshape(5,5,3))

# Set evaluation delta
surf.delta = 0.05

# Evaluate surface
surf.evaluate()

# Extract evaluated points
eval_points = surf.evalpts

# Create a new matplotlib figure and axes
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
x = [pt[0] for pt in eval_points]
y = [pt[1] for pt in eval_points]
z = [pt[2] for pt in eval_points]

# print([[a**2 + b**2 + c**2] for a, b, c in zip(x,y,z)])

ax.scatter(x, y, z, c='b', marker='o')

# Scatter plot for control points
control_points_flat_list = [pt for sublist in surf.ctrlpts2d for pt in sublist]
control_x = [pt[0] / pt[3] for pt in control_points_flat_list]  # pt[3] is the weight, using homogeneous coordinates.
control_y = [pt[1] / pt[3] for pt in control_points_flat_list]
control_z = [pt[2] / pt[3] for pt in control_points_flat_list]

ax.scatter(control_x, control_y, control_z, c='r', marker='^', label='Control Points')

# Adding a legend to differentiate between evaluated points and control points
ax.legend()

# Adding labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('NURBS Surface')

# Show the plot
plt.show()
