# NURBS functions from the provided quad_sphere.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from matplotlib import cm
import time

def N(i, k, t, knot_vector, cache):
    """
    Modified N function with boundary checks and caching.
    """
    key = (i, k, t)
    if key in cache:
        return cache[key]
    
    # Boundary checks
    if t == knot_vector[-1] and i == len(knot_vector) - k - 2:
        cache[key] = 1.0
        return 1.0
    if t == knot_vector[0] and i == 0:
        cache[key] = 1.0
        return 1.0
    
    if k == 0:
        result = 1.0 if knot_vector[i] <= t < knot_vector[i+1] else 0.0
    else:
        N1 = ((t - knot_vector[i]) / (knot_vector[i+k] - knot_vector[i])) * N(i, k-1, t, knot_vector, cache) if knot_vector[i+k] - knot_vector[i] != 0 else 0
        N2 = ((knot_vector[i+k+1] - t) / (knot_vector[i+k+1] - knot_vector[i+1])) * N(i+1, k-1, t, knot_vector, cache) if knot_vector[i+k+1] - knot_vector[i+1] != 0 else 0
        result = N1 + N2

    cache[key] = result
    return result

def R(i, j, k, l, u, v, knot_vector_u, knot_vector_v, weights, N_func):
    cache_u = {}
    cache_v = {}
    N_u = np.array([N_func(i_p, k, u, knot_vector_u, cache_u) for i_p in range(len(knot_vector_u) - k - 1)])
    N_v = np.array([N_func(j_p, l, v, knot_vector_v, cache_v) for j_p in range(len(knot_vector_v) - l - 1)])
    N_ij = np.outer(N_u, N_v)
    W_ij = weights * N_ij
    return N_ij[i, j] * weights[i, j] / np.sum(W_ij)

def surface_point(u, v, control_points, weights, knot_vector_u, knot_vector_v, N_func, R_func):
    num_points_u = control_points.shape[0]
    num_points_v = control_points.shape[1]
    k = len(knot_vector_u) - num_points_u - 1
    l = len(knot_vector_v) - num_points_v - 1

    point = np.zeros(3)
    for i in range(num_points_u):
        for j in range(num_points_v):
            Rij = R_func(i, j, k, l, u, v, knot_vector_u, knot_vector_v, weights, N_func)
            point += Rij * control_points[i, j]
    return point

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

# # Function to reverse the order of control points for specified patches
# def reverse_order_for_patches(rotation_matrix, control_points):
#     if np.array_equal(rotation_matrix, rotations["left"]):
#         # control_points = np.array([row[::-1] for row in control_points])
#         return control_points[::-1]
#     if np.array_equal(rotation_matrix, rotations["back"]):
#         # control_points = np.array([row[::-1] for row in control_points])
#         return np.array([row[::-1] for row in control_points])
#     if np.array_equal(rotation_matrix, rotations["top"]):
#         # control_points = np.array([row[::-1] for row in control_points])
#         return np.array([row[::-1] for row in control_points])
#     return control_points

# # Modified compute_surface_points function with reversed order for specified patches
# def compute_surface_points(rotation_matrix, control_points, weights, knot_vector_u, knot_vector_v, 
#                            surface_point_func=surface_point, N_func=N, R_func=R, num_points=50):
#     # Reverse the order of control points for the specified patches
#     control_points = reverse_order_for_patches(rotation_matrix, control_points)
#     rotated_control_points = np.einsum('ij,klj->kli', rotation_matrix, control_points)
#     u_values = np.linspace(0, 0.99, num_points, endpoint=True)
#     v_values = np.linspace(0, 0.99, num_points, endpoint=True)
#     surface_points = []
#     for u in u_values:
#         for v in v_values:
#             pt = surface_point_func(u, v, rotated_control_points, weights, knot_vector_u, knot_vector_v, N_func, R_func)
#             if not np.isnan(pt).any():
#                 surface_points.append(pt)
#     return np.array(surface_points)


# fig = plt.figure(figsize=(15, 10))
# ax = fig.add_subplot(111, projection='3d')

rotations = {
    "top": np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
    "front": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
    "back": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    "left": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
    "right": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
    "bottom": np.eye(3)
}

# for direction, rotation_matrix in rotations.items():
#     surface_points = compute_surface_points(rotation_matrix, actual_control_points_bottom, weights_bottom_float, knot_vector_u_bottom, knot_vector_v_bottom)
#     distances = np.linalg.norm(surface_points, axis=1)
#     max_error = np.max(np.abs(distances - 1))
#     print(f"Maximum error for {direction.capitalize()} Face:", max_error)
#     x_vals = surface_points[:, 0]
#     y_vals = surface_points[:, 1]
#     z_vals = surface_points[:, 2]
#     ax.scatter(x_vals, y_vals, z_vals, s=5, label=f'{direction.capitalize()} Face')

# ax.set_title('All Six Faces of Sphere Under Cube Topology')
# ax.legend()
# plt.show()

# Function to compute edge points
def compute_edge_points(u_or_v_value, is_u, control_points, weights, knot_vector_u, knot_vector_v, num_points=50):
    edge_points = []
    if is_u:
        u_values = [u_or_v_value] * num_points
        v_values = np.linspace(0, 0.99, num_points, endpoint=True)
    else:
        u_values = np.linspace(0, 0.99, num_points, endpoint=True)
        v_values = [u_or_v_value] * num_points

    for u, v in zip(u_values, v_values):
        pt = surface_point(u, v, control_points, weights, knot_vector_u, knot_vector_v, N, R)
        if not np.isnan(pt).any():
            edge_points.append(pt)

    return np.array(edge_points)

# Function to compute surface points with adjusted boundaries
def compute_surface_points_adjusted(rotation_matrix, control_points, weights, knot_vector_u, knot_vector_v, num_points=50):
    surface_points = []
    u_values = np.linspace(0, 1, num_points, endpoint=True)
    v_values = np.linspace(0, 1, num_points, endpoint=True)
    
    for u in u_values:
        for v in v_values:
            pt = surface_point(u, v, control_points, weights, knot_vector_u, knot_vector_v, N, R)
            if not np.isnan(pt).any():
                surface_points.append(pt)

    surface_points = np.array(surface_points)
    return np.einsum('ij,kj->ki', rotation_matrix, surface_points)

# # Collect surface points for each face with adjusted boundaries
# combined_surface_points_adjusted = []
# for direction, rotation_matrix in rotations.items():
#     surface_points_adjusted = compute_surface_points_adjusted(rotation_matrix, actual_control_points_bottom, weights_bottom_float, knot_vector_u_bottom, knot_vector_v_bottom)
#     combined_surface_points_adjusted.extend(surface_points_adjusted)

# combined_surface_points_adjusted = np.array(combined_surface_points_adjusted)

# # Visualization
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# x_vals_adjusted = combined_surface_points_adjusted[:, 0]* a
# y_vals_adjusted = combined_surface_points_adjusted[:, 1]* b
# z_vals_adjusted = combined_surface_points_adjusted[:, 2]* c

# t1 = time.time()
# print('The time it take is: ', t1 - t0)

# ax.scatter(x_vals_adjusted, y_vals_adjusted, z_vals_adjusted, s=5, label='Combined Sphere (Adjusted Boundaries)')
# ax.set_title('Combined Ellipsoid from All Faces (Adjusted Boundaries)')
# ax.legend()
# plt.show()

def random_function_space(x, y, z):
    return np.sin(5*x) + np.cos(5*y) + np.sin(5*z)

def compute_surface_points_with_function(rotation_matrix, control_points, weights, knot_vector_u, knot_vector_v, func, num_points=50):
    u_values = np.linspace(0, 1, num_points, endpoint=True)
    v_values = np.linspace(0, 1, num_points, endpoint=True)
    
    surface_points = np.zeros((num_points, num_points, 3))
    function_values = np.zeros((num_points, num_points))
    
    for i, u in enumerate(u_values):
        for j, v in enumerate(v_values):
            point = surface_point(u, v, control_points, weights, knot_vector_u, knot_vector_v, N, R)
            rotated_point = rotation_matrix @ point
            surface_points[i, j] = rotated_point
            function_values[i, j] = func(rotated_point[0], rotated_point[1], rotated_point[2])
            
    return surface_points, function_values

def compute_ellipsoid_points_with_function(rotation_matrix, control_points, weights, knot_vector_u, knot_vector_v, func, a, b, c, num_points=50):
    u_values = np.linspace(0, 1, num_points, endpoint=True)
    v_values = np.linspace(0, 1, num_points, endpoint=True)
    
    surface_points = np.zeros((num_points, num_points, 3))
    function_values = np.zeros((num_points, num_points))
    
    for i, u in enumerate(u_values):
        for j, v in enumerate(v_values):
            point = surface_point(u, v, control_points, weights, knot_vector_u, knot_vector_v, N, R)
            rotated_point = rotation_matrix @ point
            rotated_point[0] = rotated_point[0] * a
            rotated_point[1] = rotated_point[1] * b
            rotated_point[2] = rotated_point[2] * c
            surface_points[i, j] = rotated_point
            #print((rotated_point[0]/a)**2 + (rotated_point[1]/b)**2 + (rotated_point[2]/c)**2)
            function_values[i, j] = func(rotated_point[0], rotated_point[1], rotated_point[2])
            #print(rotated_point[0], rotated_point[1], rotated_point[2], function_values[i, j])
            
    return surface_points, function_values

rotations = [
    np.eye(3),
    np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),  # 90 degree rotation around y
    np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),  # -90 degree rotation around y
    np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),  # 90 degree rotation around x
    np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),  # -90 degree rotation around x
    np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),  # 180 degree rotation around y
]

a, b, c = 1, 1, 1

all_surface_points = []
all_function_values = []

for rotation in rotations:
    surface_points, function_values = compute_ellipsoid_points_with_function(rotation, actual_control_points_bottom, weights_bottom_float, knot_vector_u_bottom, knot_vector_v_bottom, random_function_space, a, b, c)
    all_surface_points.append(surface_points)
    all_function_values.append(function_values)


# Visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plotting all the surfaces
for surface_points, function_values in zip(all_surface_points, all_function_values):
    x_vals = surface_points[:, :, 0]
    y_vals = surface_points[:, :, 1]
    z_vals = surface_points[:, :, 2]
    colors = cm.viridis(function_values)
    ax.plot_surface(x_vals, y_vals, z_vals, facecolors=colors, rstride=1, cstride=1, alpha=0.6)

ax.set_title("Ellipsoid with Function Space Color Map")
plt.show()

###################################################################################################################################

# import numpy as np
# import sympy as sp
# from scipy.linalg import solve

# # Define the random function f on the surface
# def random_function_space(x, y, z):
#     return np.sin(5*x) + np.cos(5*y) + np.sin(5*z)

# # Define the Bernstein basis function
# def bernstein_basis(i, n, u):
#     return sp.binomial(n, i) * (u**i) * ((1-u)**(n-i))

# # Define the tensor Bernstein basis functions b_{ij}
# def tensor_bernstein_basis(i, j, n, u, v):
#     return bernstein_basis(i, n, u) * bernstein_basis(j, n, v)

# # Use Gaussian quadrature to compute the integrals
# def gaussian_quadrature_2D(func, n_points=3):
#     # Define Gauss quadrature points and weights for interval [0, 1]
#     points, weights = np.polynomial.legendre.leggauss(n_points)
#     points = 0.5 * (points + 1)  # Map from [-1, 1] to [0, 1]
#     weights *= 0.5  # Adjust weights for interval [0, 1]

#     integral = 0
#     for i in range(n_points):
#         for j in range(n_points):
#             u_val, v_val = points[i], points[j]
#             w_u, w_v = weights[i], weights[j]
#             integral += func(u_val, v_val) * w_u * w_v
#     return integral

# # Compute the matrix A and vector b
# n = 4
# A = np.zeros(((n+1)**2, (n+1)**2))
# b = np.zeros((n+1)**2)

# for i in range(n+1):
#     for j in range(n+1):
#         for k in range(n+1):
#             for l in range(n+1):
#                 # Define the integrand for the element A_ij,kl
#                 def integrand_A(u, v):
#                     return tensor_bernstein_basis(i, j, n, u, v) * tensor_bernstein_basis(k, l, n, u, v)
                
#                 A[i*(n+1) + j, k*(n+1) + l] = gaussian_quadrature_2D(integrand_A)
                
#                 # Define the integrand for the element b_kl
#                 def integrand_b(u, v):
#                     x, y, z = surface_point(u, v, control_points_float, weights_float, knot_vector_u_bottom, knot_vector_v_bottom, N, R)
#                     return random_function_space(x, y, z) * tensor_bernstein_basis(k, l, n, u, v)
                
#                 b[k*(n+1) + l] = gaussian_quadrature_2D(integrand_b)

# # Solve the linear system A * beta = b
# beta = solve(A, b)

# # Reshape beta to get the coefficients beta_ij
# beta_ij = beta.reshape((n+1, n+1))
