import matplotlib.pyplot as plt
import numpy as np

# Monkey-patching np.float
if not hasattr(np, 'float'):
    np.float = float
from geomdl import BSpline
from geomdl import utilities
from geomdl import operations
from geomdl import helpers

# Create a B-spline curve instance
curve = BSpline.Curve()

# Set up the B-spline curve
curve.degree = 2  # Quadratic B-spline curve
curve.ctrlpts = [[0, 0], [3, 4], [6, -1], [10, 0]]  # Control points
curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))  # Generate knot vector

# Function to evaluate B-spline basis function
def bspline_basis(i, k, t, knot_vector):
    if k == 0:
        return 1.0 if knot_vector[i] <= t < knot_vector[i + 1] else 0.0
    else:
        coeff1 = ((t - knot_vector[i]) / (knot_vector[i + k] - knot_vector[i]) if knot_vector[i + k] - knot_vector[i] != 0 else 0) * bspline_basis(i, k - 1, t, knot_vector)
        coeff2 = ((knot_vector[i + k + 1] - t) / (knot_vector[i + k + 1] - knot_vector[i + 1]) if i + k + 1 < len(knot_vector) and knot_vector[i + k + 1] - knot_vector[i + 1] != 0 else 0) * bspline_basis(i + 1, k - 1, t, knot_vector)
        
        return coeff1 + coeff2

# Function to plot each basis function of the B-spline
def plot_basis_functions(curve, num_points=1000):
    u_vals = np.linspace(0, 1, num_points)
    n = len(curve.ctrlpts) - 1  # n is the index of the last control point
    p = curve.degree  # Degree of the B-spline
    
    plt.figure(figsize=(10,5))
    
    for i in range(n + p):
        basis_vals = [bspline_basis(i, p, u, curve.knotvector) for u in u_vals]
        plt.plot(u_vals, basis_vals, label=f'N_{i},{p}')
        
    plt.title('B-spline Basis Functions')
    plt.xlabel('u')
    plt.ylabel('Basis Function Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# Plot basis functions before knot insertion
plot_basis_functions(curve)

# Insert a knot at u=0.5
operations.insert_knot(curve, [0.5], 1)

# Plot basis functions after knot insertion
plot_basis_functions(curve)