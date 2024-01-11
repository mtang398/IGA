# NURBS functions from the provided quad_sphere.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from matplotlib import cm
import time
from scipy.linalg import solve
from scipy.special import comb
import nurbs_c
from nurbs_c import ffi, lib

# Monkey-patching np.float
if not hasattr(np, 'float'):
    np.float = float

from geomdl import operations
from geomdl import NURBS
from geomdl.visualization import VisMPL
# from gaussian_quadrature_2D import gaussian_quadrature_2D

def surface_point_cffi(u, v, control_points, weights, knot_vector_u, knot_vector_v):
    num_points_u, num_points_v, _ = control_points.shape
    control_points_flattened = control_points.ravel()
    weights_flattened = weights.ravel()

    # Convert data to CFFI compatible types
    # Convert data to CFFI compatible types
    c_control_points = ffi.new("double[]", control_points_flattened.tolist())
    c_weights = ffi.new("double[]", weights_flattened.tolist())
    c_knot_vector_u = ffi.new("double[]", knot_vector_u)
    c_knot_vector_v = ffi.new("double[]", knot_vector_v)
    point = ffi.new("double[3]")

    # Call the C function
    lib.surface_point(u, v, c_control_points, num_points_u, num_points_v, c_weights, len(weights_flattened), c_knot_vector_u, len(knot_vector_u), c_knot_vector_v, len(knot_vector_v), point)

    # Convert the result back to a numpy array
    return np.array([point[0], point[1], point[2]])

# Define the random function f on the surface
def random_function_space(x, y, z):
    return np.sin(5*x) + np.cos(5*y) + np.sin(5*z)

def tensor_bernstein_basis_cffi(i, j, n, u_array, v_array):
    # Check if the inputs are scalars or arrays
    u_is_scalar = np.isscalar(u_array)
    v_is_scalar = np.isscalar(v_array)
    
    # Convert scalars to one-element arrays for uniformity
    if u_is_scalar:
        u_array = np.array([u_array])
    if v_is_scalar:
        v_array = np.array([v_array])
    
    # Get array sizes
    u_size = len(u_array)
    v_size = len(v_array)
    
    # Create C-compatible arrays
    u_array_c = ffi.new("double[]", u_array.tolist())
    v_array_c = ffi.new("double[]", v_array.tolist())
    
    # Create an empty result array to store the output
    result_c = ffi.new("double * [%d]" % u_size)
    for u in range(u_size):
        result_c[u] = ffi.new("double[]", v_size)
    
    # Call the CFFI function
    lib.tensor_bernstein_basis(i, j, n, u_array_c, v_array_c, result_c, u_size, v_size)
    
    # Convert the result back to a NumPy array
    result = np.zeros((u_size, v_size))
    for u in range(u_size):
        for v in range(v_size):
            result[u, v] = result_c[u][v]
    
    # If the input was a scalar, return a scalar
    if u_is_scalar and v_is_scalar:
        return result[0, 0]
    elif u_is_scalar:
        return result[0, :]
    elif v_is_scalar:
        return result[:, 0]
    
    return result

# Use Gaussian quadrature to compute the integrals
def gaussian_quadrature_2D(func, n_points=20):
    # Define Gauss quadrature points and weights for interval [0, 1]
    points, weights = np.polynomial.legendre.leggauss(n_points)
    points = 0.5 * (points + 1)  # Map from [-1, 1] to [0, 1]
    weights *= 0.5  # Adjust weights for interval [0, 1]
    
    # Generate 2D grid of points and weights
    u_vals, v_vals = np.meshgrid(points, points)
    w_u, w_v = np.meshgrid(weights, weights)
    
    # Calculate function values for the entire grid
    func_vals = np.vectorize(func)(u_vals, v_vals)
    
    # Calculate the integral using vectorized operations
    integral = np.sum(func_vals * w_u * w_v)
    
    return integral

t0 = time.time()
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


# Compute the matrix A and vector b
start_compute = time.time()

# Order
p = 5
# h_ref = 0
# k_ref = 0

# # Combine control_points and weights
# control_points_4d = np.concatenate([control_points_bottom_float, weights_bottom_float[..., None]], axis=-1)

# # Create a NURBS surface object
# surf = NURBS.Surface()

# # Setting the degrees and knot vectors
# surf.degree_u = 4  # Degree in the u direction
# surf.degree_v = 4  # Degree in the v direction

# # Set the weighted control points
# surf.ctrlpts2d = control_points_4d.tolist()

# surf.knotvector_u = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
# surf.knotvector_v = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# def perform_insert_knot(surf, h_ref_order):
#     if h_ref_order == 0:
#         # Do nothing
#         return
    
#     num_insertions = 4  # number of times the knot to be inserted
#     interval = 1 / (2 ** h_ref_order)  # interval directly calculated from h_ref_order
    
#     # Insert knots at intervals
#     for insertion_point in np.arange(interval, 1, interval):
#         operations.insert_knot(surf, [insertion_point, insertion_point], [num_insertions, num_insertions])

# def perform_insert_knot_k(surf, k_ref_order):
#     if k_ref_order == 0:
#         # Do nothing
#         return
    
#     num_insertions = 1  # number of times the knot to be inserted
#     interval = 1 / (2 ** k_ref_order)  # interval directly calculated from h_ref_order
    
#     # Insert knots at intervals
#     for insertion_point in np.arange(interval, 1, interval):
#         operations.insert_knot(surf, [insertion_point, insertion_point], [num_insertions, num_insertions])


# perform_insert_knot(surf, h_ref)
# perform_insert_knot_k(surf, k_ref)

# if k_ref != 0: 
#     actual_control_points_bottom = np.array(surf._control_points)[..., :3].reshape(4+2**k_ref, 4+2**k_ref, 3).astype(float)
#     weights_bottom_float = np.array(surf._control_points)[..., 3].reshape(4+2**k_ref, 4+2**k_ref).astype(float)
#     actual_control_points_bottom = compute_actual_control_points(actual_control_points_bottom, weights_bottom_float, 1)

#     knot_vector_u_bottom = surf.knotvector_u
#     knot_vector_v_bottom = surf.knotvector_v

#     p = p + k_ref    

# if h_ref != 0:
#     actual_control_points_bottom = np.array(surf._control_points)[..., :3].reshape(4*2**h_ref+1, 4*2**h_ref+1, 3).astype(float)
#     weights_bottom_float = np.array(surf._control_points)[..., 3].reshape(4*2**h_ref+1, 4*2**h_ref+1).astype(float)
#     actual_control_points_bottom = compute_actual_control_points(actual_control_points_bottom, weights_bottom_float, 1)

#     knot_vector_u_bottom = surf.knotvector_u
#     knot_vector_v_bottom = surf.knotvector_v

    # print('The current Control Points is: ', actual_control_points_bottom)
    # print('The current weights is: ', weights_bottom_float)

def compute_A_and_b(n, 
                    tensor_bernstein_basis_cffi, 
                    gaussian_quadrature_2D, 
                    surface_point_cffi, 
                    actual_control_points_bottom, 
                    weights_bottom_float, 
                    knot_vector_u_bottom, 
                    knot_vector_v_bottom, 
                    random_function_space):
    
    A = np.zeros(((n+1)**2, (n+1)**2))
    b = np.zeros((n+1)**2)
    
    for i in range(n+1):
        for j in range(n+1):
            for k in range(n+1):
                for l in range(n+1):
                    # Define the integrand for the element A_ij,kl
                    def integrand_A(u, v):
                        return tensor_bernstein_basis_cffi(i, j, n, u, v) * tensor_bernstein_basis_cffi(k, l, n, u, v)
                    
                    A[i*(n+1) + j, k*(n+1) + l] = gaussian_quadrature_2D(integrand_A)
                    
                    # Define the integrand for the element b_kl
                    def integrand_b(u, v):
                        x, y, z = surface_point_cffi(u, v, actual_control_points_bottom, weights_bottom_float, knot_vector_u_bottom, knot_vector_v_bottom)
                        return random_function_space(x, y, z) * tensor_bernstein_basis_cffi(k, l, n, u, v)
                    
                    b[k*(n+1) + l] = gaussian_quadrature_2D(integrand_b)
    
    return A, b, n

A, b, n = compute_A_and_b(p, 
                        tensor_bernstein_basis_cffi, 
                        gaussian_quadrature_2D, 
                        surface_point_cffi, 
                        actual_control_points_bottom, 
                        weights_bottom_float, 
                        knot_vector_u_bottom, 
                        knot_vector_v_bottom, 
                        random_function_space)
                
end_compute = time.time()
print("Time spent on compute the matrix A and vector b: ", end_compute - start_compute)

from scipy.sparse.linalg import gmres
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spilu
from scipy.sparse.linalg import LinearOperator

# Using LU decomposition
from scipy.linalg import lu_factor, lu_solve

# startLU = time.time()
lu, piv = lu_factor(A)
beta_ij_vectorized = lu_solve((lu, piv), b)

# Reshape the solution to the matrix form
beta_ij = beta_ij_vectorized.reshape((n+1, n+1))
# endLU = time.time()
# print("Time used on LU decomposition: ", startLU - endLU)


# Generate a mesh of points on the NURBS sphere surface
u_values = np.linspace(0, 1, 50)
v_values = np.linspace(0, 1, 50)
U, V = np.meshgrid(u_values, v_values)

start_evaluation = time.time()
# Evaluate the NURBS surface at these points
X = np.zeros_like(U)
Y = np.zeros_like(V)
Z = np.zeros_like(U)
for i in range(U.shape[0]):
    for j in range(U.shape[1]):
        X[i, j], Y[i, j], Z[i, j] = surface_point_cffi(U[i, j], V[i, j], actual_control_points_bottom, weights_bottom_float, knot_vector_u_bottom, knot_vector_v_bottom)
end_evaluation = time.time()

print("Time spent on evaluate the NURBS surface at these points: ", end_evaluation - start_evaluation)

start_F = time.time()
# Evaluate the actual function f and our approximation F over this mesh
F_values = np.zeros_like(U)
f_values = np.zeros_like(U)
for i in range(U.shape[0]):
    for j in range(U.shape[1]):
        sum_val = 0
        for p in range(n+1):
            for q in range(n+1):
                sum_val += beta_ij[p, q] * tensor_bernstein_basis_cffi(p, q, n, U[i, j], V[i, j])
        F_values[i, j] = sum_val
        f_values[i, j] = random_function_space(X[i, j], Y[i, j], Z[i, j])
        
# Assuming F_values and f_values are your 2D arrays
l2_error = np.sqrt(np.sum((F_values - f_values)**2))
print("The l2 error of F approximating f is: ", l2_error)
        
end_F = time.time()
print("Time spent on evaluate the actual function f and our approximation F over this mesh: ", end_F - start_F)
        
print('The Maximum Error is: ', max([abs(F_values[i,j] - f_values[i,j]) for i in range(U.shape[0]) for j in range(U.shape[1])]))
#print([abs(F_values[i,j] - f_values[i,j]) for i in range(U.shape[0]) for j in range(U.shape[1])])

t1 = time.time()

print('The time spent is: ', t1 - t0)

def plot_functions_side_by_side(X, Y, Z, f_values, F_values):
    fig = plt.figure(figsize=(18, 9))

    # Create a custom colormap for F values
    cmap = plt.cm.viridis
    norm = plt.Normalize(F_values.min(), F_values.max())
    color_mapper = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    color_mapper.set_array([])

    # Plot the actual function f
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z, facecolors=cmap(norm(f_values)), rstride=1, cstride=1, alpha=0.8)
    ax1.set_title("Actual Function f on NURBS Sphere")
    ax1.view_init(elev=30, azim=40)
    colorbar1 = fig.colorbar(color_mapper, ax=ax1, pad=0.05, orientation='vertical')
    colorbar1.set_label('f values')
    
    # Plot the approximation F
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z, facecolors=cmap(norm(F_values)), rstride=1, cstride=1, alpha=0.8)
    ax2.set_title("Approximation F on NURBS Sphere")
    ax2.view_init(elev=30, azim=40)
    colorbar2 = fig.colorbar(color_mapper, ax=ax2, pad=0.05, orientation='vertical')
    colorbar2.set_label('F values')

    # Calculate the relative error between f and F
    relative_error = np.abs((f_values - F_values) / f_values)
    
    # Normalize the color mapping for relative error
    rel_error_norm = plt.Normalize(relative_error.min(), relative_error.max())
    print(relative_error.min(), relative_error.max())
    color_mapper_rel_error = plt.cm.ScalarMappable(cmap=cmap, norm=rel_error_norm)
    color_mapper_rel_error.set_array([])

    # Plot the relative error graph
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X, Y, Z, facecolors=cmap(rel_error_norm(relative_error)), rstride=1, cstride=1, alpha=0.8)
    ax3.set_title("Relative Error between f and F")
    ax3.view_init(elev=30, azim=40)
    colorbar3 = fig.colorbar(color_mapper_rel_error, ax=ax3, pad=0.05, orientation='vertical')
    colorbar3.set_label('Relative Error')
    
    plt.tight_layout()
    plt.show()
    
# Call the plotting function
plot_functions_side_by_side(X, Y, Z, f_values, F_values)
t2 = time.time()
print('The time spent with graphing is: ', t2 - t0)