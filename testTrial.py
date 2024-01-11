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

from geomdl import NURBS
from geomdl.visualization import VisMPL
from geomdl import operations
# from gaussian_quadrature_2D import gaussian_quadrature_2D

def generate_subgrids(u_range, v_range, n):
    """
    Generate 2^n x 2^n subgrids for a given (u, v) grid range.
    
    Parameters:
        u_range (tuple): Range for u parameter (u_min, u_max)
        v_range (tuple): Range for v parameter (v_min, v_max)
        n (int): Level of refinement, resulting in 2^n x 2^n subgrids
        
    Returns:
        list: A list of tuples representing the (u, v) ranges for each subgrid
    """
    u_min, u_max = u_range
    v_min, v_max = v_range
    
    # Calculate step size for subgrids
    u_step = (u_max - u_min) / (2 ** n)
    v_step = (v_max - v_min) / (2 ** n)
    
    subgrids = []
    
    # Generate subgrids
    for i in range(2 ** n):
        for j in range(2 ** n):
            u_sub_min = u_min + i * u_step
            u_sub_max = u_min + (i + 1) * u_step
            v_sub_min = v_min + j * v_step
            v_sub_max = v_min + (j + 1) * v_step
            subgrids.append(((u_sub_min, u_sub_max), (v_sub_min, v_sub_max)))
            
    return subgrids

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

from scipy.special import spherical_jn, spherical_yn
from numpy.polynomial.legendre import legval  # To compute the Legendre polynomial
from scipy.special import spherical_jn, spherical_yn, lpmv  # lpmv for vectorized Legendre polynomial computation

# Define the random function f on the surface
def plane_wave_impinge_on_sphere(x, y, z, a=1.0, k=10.0, N=50):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r) if r != 0 else 0
    
    n_values = np.arange(1, N+1)
    
    # Precompute spherical Bessel functions and their derivatives
    jn_values = spherical_jn(n_values, k*a)
    jn_prime_values = spherical_jn(n_values, k*a, derivative=True)
    yn_values = spherical_yn(n_values, k*a)
    yn_prime_values = spherical_yn(n_values, k*a, derivative=True)
    
    hn_prime_values = jn_prime_values + 1j * yn_prime_values
    
    # Precompute Legendre polynomials
    cos_theta = np.cos(theta)
    Pn_values = lpmv(0, n_values, cos_theta)
    
    # Precompute spherical Hankel function of the first kind for all n
    h_n_values = spherical_jn(n_values, k*r) + 1j * spherical_yn(n_values, k*r)
    
    # Compute the term for all n and sum them up
    terms = 1j**n_values * (2*n_values + 1) * (jn_prime_values / hn_prime_values) * Pn_values * h_n_values
    phi_s = np.sum(terms)
        
    return phi_s.real

def tensor_bspline_basis(i, j, k, l, u, v, knot_vector_u, knot_vector_v):
    # Convert Python list to C array
    c_knot_vector_u = ffi.new("double[]", knot_vector_u)
    c_knot_vector_v = ffi.new("double[]", knot_vector_v)
    
    weights_size = (len(knot_vector_u) - k - 1) * (len(knot_vector_v) - l - 1)
    c_weights = ffi.new("double[]", [1.0] * weights_size)  # assuming uniform weights of 1
    
    cache_count_u = ffi.new("int *", 0)
    cache_count_v = ffi.new("int *", 0)
    cache_u = ffi.new("CacheItem[]", weights_size)  # Use weights_size as a sufficient large size for cache
    cache_v = ffi.new("CacheItem[]", weights_size)
    
    result = lib.R(i, j, k, l, u, v,
                 c_knot_vector_u, len(knot_vector_u),
                 c_knot_vector_v, len(knot_vector_v),
                 c_weights, weights_size,
                 cache_u, cache_count_u,
                 cache_v, cache_count_v)
    
    return result

# Use Gaussian quadrature to compute the integrals
def gaussian_quadrature_2D(func, u_range, v_range, n_points=20):
    # print(u_range, v_range)
    # Define Gauss quadrature points and weights for interval [0, 1]
    points, weights = np.polynomial.legendre.leggauss(n_points)
    
    # Map points to the respective u and v ranges
    u_min, u_max = u_range
    v_min, v_max = v_range
    points_u = u_min + 0.5 * (u_max - u_min) * (points + 1)
    points_v = v_min + 0.5 * (v_max - v_min) * (points + 1)
    
    # Adjust weights for the new interval
    weights_u = 0.5 * weights * (u_max - u_min)
    weights_v = 0.5 * weights * (v_max - v_min)
    
    # Generate 2D grid of points and weights
    u_vals, v_vals = np.meshgrid(points_u, points_v)
    w_u, w_v = np.meshgrid(weights_u, weights_v)
    
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

# knot_vector_u_h_refined = [0, 0, 0, 0 ,0, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1]
# knot_vector_u_h_refined = [0, 0, 0, 0 ,0, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1]

def compute_actual_control_points(weighted_control_points, weights, radius):
    return weighted_control_points / weights[..., None] * radius

# Compute the matrix A and vector b
start_compute = time.time()

# Order
p = 4
k_ref = 0
h_ref = 0
h_check = 0
n_refinement = 0 # Level of refinement, can be set to any value between 0 and 8

if h_ref != 0:
    h_check = 1

if k_ref != 0:
    p = p + k_ref

basis_knot_u = [0]*(p+1) + [1]*(p+1)
basis_knot_v = [0]*(p+1) + [1]*(p+1)

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

surf.knotvector_u = knot_vector_u_bottom
surf.knotvector_v = knot_vector_v_bottom

def perform_insert_knot(surf, h_ref_order):
    if h_ref_order == 0:
        # Do nothing
        return
    
    num_insertions = 1  # number of times the knot to be inserted
    interval = 1 / (2 ** h_ref_order)  # interval directly calculated from h_ref_order
    
    # Insert knots at intervals
    for insertion_point in np.arange(interval, 1, interval):
        operations.insert_knot(surf, [insertion_point, insertion_point], [num_insertions, num_insertions])

def perform_insert_knot_4_h_refine(surf, h_ref_order, p):
    if h_ref_order == 0:
        # Do nothing
        return
    
    num_insertions = p  # number of times the knot to be inserted
    interval = 1 / (2 ** h_ref_order)  # interval directly calculated from h_ref_order
    
    # Insert knots at intervals
    for insertion_point in np.arange(interval, 1, interval):
        operations.insert_knot(surf, [insertion_point, insertion_point], [num_insertions, num_insertions])

def simple_knot_add(k_ref):
    num_insertion = 1
    interval = 1 / (2 ** k_ref)
    insert = []

    for insertion_point in np.arange(interval, 1, interval):
        insert.append(insertion_point)

    return insert

def simple_knot_add_4_h_refinement(k_ref, p):
    interval = 1 / (2 ** k_ref)
    insert = []

    for insertion_point in np.arange(interval, 1, interval):
        for _ in range(p):
            insert.append(insertion_point)

    return insert

perform_insert_knot(surf, k_ref)
perform_insert_knot_4_h_refine(surf, h_ref, p)

cp_bottom = actual_control_points_bottom
w_bottom = weights_bottom_float

if k_ref != 0:
    cp_bottom = np.array(surf._control_points)[..., :3].reshape(4 + 2**k_ref + p*h_check*2**(2**h_ref - 1), 4 + 2**k_ref + p*h_check*2**(2**h_ref - 1), 3).astype(float)
    w_bottom = np.array(surf._control_points)[..., 3].reshape(4+2**k_ref, 4+2**k_ref).astype(float)
    cp_bottom = compute_actual_control_points(cp_bottom, w_bottom, 1)

    knots = simple_knot_add(k_ref)

    basis_knot_u = [0]*(p+1) + knots + [1]*(p+1)
    basis_knot_v = [0]*(p+1) + knots + [1]*(p+1)

if h_ref != 0:
    cp_bottom = np.array(surf._control_points)[..., :3].reshape(4 + 2**k_ref + p*h_check*(2**h_ref - 1), 4 + 2**k_ref + p*h_check*(2**h_ref - 1), 3).astype(float)
    w_bottom = np.array(surf._control_points)[..., 3].reshape(4 + 2**k_ref + p*h_check*(2**h_ref - 1), 4 + 2**k_ref + p*h_check*(2**h_ref - 1)).astype(float)
    cp_bottom = compute_actual_control_points(cp_bottom, w_bottom, 1)

    knots = simple_knot_add_4_h_refinement(h_ref, p)

    basis_knot_u = [0]*(p+1) + knots + [1]*(p+1)
    basis_knot_v = [0]*(p+1) + knots + [1]*(p+1)

# def compute_A_and_b(n, 
#                     tensor_bernstein_basis_cffi, 
#                     gaussian_quadrature_2D, 
#                     surface_point_cffi, 
#                     actual_control_points_bottom, 
#                     weights_bottom_float, 
#                     knot_vector_u_bottom, 
#                     knot_vector_v_bottom, 
#                     random_function_space,
#                     u_interval,
#                     v_interval):
    
#     A = np.zeros(((n+1)**2, (n+1)**2))
#     b = np.zeros((n+1)**2)
    
#     for i in range(n+1):
#         for j in range(n+1):
#             for k in range(n+1):
#                 for l in range(n+1):
#                     # Define the integrand for the element A_ij,kl
#                     def integrand_A(u, v):
#                         return tensor_bernstein_basis_cffi(i, j, n, u, v) * tensor_bernstein_basis_cffi(k, l, n, u, v)
                    
#                     A[i*(n+1) + j, k*(n+1) + l] = gaussian_quadrature_2D(integrand_A, u_interval, v_interval)
                    
#                     # Define the integrand for the element b_kl
#                     def integrand_b(u, v):
#                         x, y, z = surface_point_cffi(u, v, actual_control_points_bottom, weights_bottom_float, knot_vector_u_bottom, knot_vector_v_bottom)
#                         return random_function_space(x, y, z) * tensor_bernstein_basis_cffi(k, l, n, u, v)
                    
#                     b[k*(n+1) + l] = gaussian_quadrature_2D(integrand_b, u_interval, v_interval)
    
#     return A, b, n

import numpy as np
import concurrent.futures

def compute_partition_for_A(i, j, k, l, n, tensor_bspline_basis, gaussian_quadrature_2D, u_interval, v_interval, knot_vector_u, knot_vector_v, p, q):
    def integrand_A(u, v):
        return (tensor_bspline_basis(i, j, p, q, u, v, knot_vector_u, knot_vector_v) * 
                tensor_bspline_basis(k, l, p, q, u, v, knot_vector_u, knot_vector_v))
    
    return i*(n+ 2**k_ref + p*h_check*(2**h_ref - 1)) + j, k*(n+ 2**k_ref + p*h_check*(2**h_ref - 1)) + l, gaussian_quadrature_2D(integrand_A, u_interval, v_interval)

def compute_partition_for_b(k, l, n, tensor_bspline_basis, gaussian_quadrature_2D, surface_point_cffi, cp, w, knot_vector_u_bottom, knot_vector_v_bottom, u_interval, v_interval, knot_vector_u, knot_vector_v, p, q):
    def integrand_b(u, v):
        x, y, z = surface_point_cffi(u, v, cp, w, knot_vector_u_bottom, knot_vector_v_bottom)
        return plane_wave_impinge_on_sphere(x, y, z) * tensor_bspline_basis(k, l, p, q, u, v, knot_vector_u, knot_vector_v)

    return k*(n+ 2**k_ref + p*h_check*(2**h_ref - 1)) + l, gaussian_quadrature_2D(integrand_b, u_interval, v_interval)

def compute_A_and_b_spline(n, tensor_bspline_basis, gaussian_quadrature_2D, surface_point_cffi, cp, w, knot_vector_u_bottom, knot_vector_v_bottom, random_function_space, u_interval, v_interval, knot_vector_u, knot_vector_v, p, q):
    A = np.zeros(((n+ 2**k_ref + p*h_check*(2**h_ref - 1))**2, (n+ 2**k_ref + p*h_check*(2**h_ref - 1))**2))
    b = np.zeros((n+ 2**k_ref + p*h_check*(2**h_ref - 1))**2)

    range_val = range(n+ 2**k_ref + p*h_check*(2**h_ref - 1))

    # Compute for A
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range_val:
            for j in range_val:
                for k in range_val:
                    for l in range_val:
                        idx_i, idx_j, value = executor.submit(compute_partition_for_A, i, j, k, l, n, tensor_bspline_basis, gaussian_quadrature_2D, u_interval, v_interval, knot_vector_u, knot_vector_v, p, q).result()
                        A[idx_i, idx_j] = value
    
    # Compute for b
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for k in range_val:
            for l in range_val:
                idx, value = executor.submit(compute_partition_for_b, k, l, n, tensor_bspline_basis, gaussian_quadrature_2D, surface_point_cffi, cp, w, knot_vector_u_bottom, knot_vector_v_bottom, u_interval, v_interval, knot_vector_u, knot_vector_v, p, q).result()
                b[idx] = value

    return A, b, n

from scipy.sparse.linalg import gmres
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spilu
from scipy.sparse.linalg import LinearOperator

# Using LU decomposition
from scipy.linalg import lu_factor, lu_solve

# Generate subgrids based on refinement level n
u_range = (0, 1)  # Original u range
v_range = (0, 1)  # Original v range

subgrids = generate_subgrids(u_range, v_range, n_refinement)

# Initialize a list to store the beta_ij for each subgrid
local_beta_ij_list = []

# Initialize global A and b
global_A_size = ((p+ 2**k_ref + p*h_check*(2**h_ref - 1)) ** 2) * len(subgrids)  # The size would depend on how you assemble the subgrids
global_A = np.zeros((global_A_size, global_A_size))
global_b = np.zeros(global_A_size)

# Counter for global A and b
counter = 0

# Loop over each subgrid
for u_sub_range, v_sub_range in subgrids:
    # print(u_sub_range, v_sub_range)
    print(weights_bottom_float)
    print(actual_control_points_bottom)
    # Compute A and b for this subgrid
    A, b, n = compute_A_and_b_spline(p, 
                              tensor_bspline_basis, 
                              gaussian_quadrature_2D, 
                              surface_point_cffi, 
                              actual_control_points_bottom, 
                              weights_bottom_float, 
                              knot_vector_u_bottom, 
                              knot_vector_v_bottom, 
                              plane_wave_impinge_on_sphere,
                              u_sub_range,
                              v_sub_range, 
                              basis_knot_u,  # New knot vector for u
                              basis_knot_v,  # New knot vector for v
                              p,  # degree in u direction
                              p)

    # print(A)
    # plt.imshow(A, cmap='viridis', interpolation='none')
    # plt.colorbar()
    # plt.show()

    # Insert this A and b into the global A and b
    size = (p+ 2**k_ref + p*h_check*(2**h_ref - 1)) ** 2
    global_A[counter:counter+size, counter:counter+size] = A
    global_b[counter:counter+size] = b
    
    # Solve the local system for this subgrid
    lu, piv = lu_factor(A)
    local_beta_ij = lu_solve((lu, piv), b)
    
    # Append this solution to the list
    local_beta_ij_list.append(local_beta_ij)

    # print(local_beta_ij)
    
    counter += size
                
end_compute = time.time()
print("Time spent on compute the matrix A and vector b: ", end_compute - start_compute)

# Concatenate to form the global beta_ij_vectorized
beta_ij_vectorized = np.concatenate(local_beta_ij_list)

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

def find_subgrid(u, v, subgrids):
    """ 
    Find which subgrid a given (u, v) point belongs to and return its index.
    """
    for idx, (u_range, v_range) in enumerate(subgrids):
        u_min, u_max = u_range
        v_min, v_max = v_range
        if u_min <= u <= u_max and v_min <= v <= v_max:
            return u_range, v_range, idx
    return None, None, -1  # Return None if the point doesn't belong to any subgrid

# In the computation of F_values and f_values:
for i in range(U.shape[0]):
    for j in range(U.shape[1]):
        
        # Identify which subgrid this (u, v) point belongs to
        u_sub_range, v_sub_range, subgrid_idx = find_subgrid(U[i, j], V[i, j], subgrids)
        
        # Extract the local beta_ij for this subgrid using the subgrid index
        size = (p + 2**k_ref + p*h_check*(2**h_ref - 1)) ** 2
        start_idx = subgrid_idx * size
        local_beta_ij = beta_ij_vectorized[start_idx:start_idx+size].reshape((p + 2**k_ref + p*h_check*(2**h_ref - 1), p + 2**k_ref + p*h_check*(2**h_ref - 1)))
        
        sum_val = 0
        for a in range(n + 2**k_ref + p*h_check*(2**h_ref - 1)):
            for b in range(n + 2**k_ref + p*h_check*(2**h_ref - 1)):
                sum_val += local_beta_ij[a, b] * tensor_bspline_basis(a, b, n, n, U[i, j], V[i,j], basis_knot_u, basis_knot_v)
        
        F_values[i, j] = sum_val
        f_values[i, j] = plane_wave_impinge_on_sphere(X[i, j], Y[i, j], Z[i, j])
        
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

    # Find the indices of the maximum relative error
    max_error_index = np.unravel_index(np.argmax(relative_error, axis=None), relative_error.shape)

    # Get the corresponding f and F values
    f = f_values[max_error_index]
    F = F_values[max_error_index]

    print("f value is ", f, ', and F value is ', F)
    
    l2_norm_relative_error = np.sqrt(np.mean(np.square(relative_error)))
    print(f'The relative l2 norm is: {l2_norm_relative_error}')
    print(f'The relative linf norm is: {relative_error.max()}')
    
    # Normalize the color mapping for relative error
    rel_error_norm = plt.Normalize(relative_error.min(), relative_error.max())
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

# # Step 2: Compute the error matrix
# error_matrix = np.abs((f_values - F_values) / f_values)
                      
# # Step 3: Generate the 3D plot
# x = np.arange(50)
# y = np.arange(50)
# X, Y = np.meshgrid(x, y)
# Z = error_matrix

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# # Add color bar
# fig.colorbar(surf)

# # Labels
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Error')

# # Show plot
# plt.show()