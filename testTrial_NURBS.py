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

# Define the random function f on the surface
def random_function_space(x, y, z):
    return np.sin(5*x) + np.cos(5*y) + np.sin(5*z)

# Define the Bernstein basis function
def bernstein_basis(i, n, u_array):
    """
    Vectorized computation of the Bernstein basis over an array of u values.
    Computes the binomial coefficient internally.
    """
    binomial_coeff = comb(n, i)
    return binomial_coeff * (u_array**i) * ((1-u_array)**(n-i))

# Define the tensor Bernstein basis functions b_{ij}
def tensor_bernstein_basis(i, j, n, u_array, v_array):
    """
    Vectorized computation of the tensor Bernstein basis over arrays of u and v values.
    """
    bernstein_u = bernstein_basis(i, n, u_array)
    bernstein_v = bernstein_basis(j, n, v_array)
    return np.outer(bernstein_u, bernstein_v)

# Use Gaussian quadrature to compute the integrals
def gaussian_quadrature_2D(func, n_points=10):
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

# Manually set n
n = 4

kv_u_bottom = [0] * (n + 1) + [1] * (n + 1)
kv_v_bottom = [0] * (n + 1) + [1] * (n + 1)

# Create an nxn matrix filled with 1s
weights_bottom = np.ones((n+1, n+1))

# Copy the values from weights_bottom_float to the top-left corner of the new matrix
weights_bottom[:weights_bottom_float.shape[0], :weights_bottom_float.shape[1]] = weights_bottom_float

# Initialize the matrix A and vector b
A = np.zeros(((n+1)**2, (n+1)**2))
b = np.zeros((n+1)**2)

# Determine k and l from knot vectors and number of control points
num_points_u = (n+1)
num_points_v = (n+1)
k = len(knot_vector_u_bottom) - num_points_u - 1
l = len(knot_vector_v_bottom) - num_points_v - 1

k1 = len(kv_u_bottom) - num_points_u - 1
l1 = len(kv_v_bottom) - num_points_v - 1

# Loop through to fill in A and b
for i in range(n+1):
    for j in range(n+1):
        for m in range(n+1):
            for nn in range(n+1):
                # Define the integrand for the element A_ij,kl
                def integrand_A(u, v):
                    Rij = R(i, j, k1, l1, u, v, kv_u_bottom, kv_v_bottom, weights_bottom, N)
                    Rmn = R(m, nn, k1, l1, u, v, kv_u_bottom, kv_v_bottom, weights_bottom, N)
                    return Rij * Rmn
                
                A[i*(n+1) + j, m*(n+1) + nn] = gaussian_quadrature_2D(integrand_A)
                
                # Define the integrand for the element b_kl
                def integrand_b(u, v):
                    x, y, z = surface_point(u, v, actual_control_points_bottom, weights_bottom_float, knot_vector_u_bottom, knot_vector_v_bottom, N, R)
                    return random_function_space(x, y, z) * R(m, nn, k1, l1, u, v, kv_u_bottom, kv_v_bottom, weights_bottom, N)
                
                b[m*(n+1) + nn] = gaussian_quadrature_2D(integrand_b)


from scipy.sparse.linalg import gmres
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spilu
from scipy.sparse.linalg import LinearOperator

# Using LU decomposition
from scipy.linalg import lu_factor, lu_solve

lu, piv = lu_factor(A)
beta_ij_vectorized = lu_solve((lu, piv), b)

# Reshape the solution to the matrix form
beta_ij = beta_ij_vectorized.reshape((n+1, n+1))


# Generate a mesh of points on the NURBS surface
u_values = np.linspace(0, 1, 50)
v_values = np.linspace(0, 1, 50)
U, V = np.meshgrid(u_values, v_values)

# Evaluate the NURBS surface at these points
X = np.zeros_like(U)
Y = np.zeros_like(V)
Z = np.zeros_like(U)

for i in range(U.shape[0]):
    for j in range(U.shape[1]):
        X[i, j], Y[i, j], Z[i, j] = surface_point(U[i, j], V[i, j], actual_control_points_bottom, weights_bottom_float, knot_vector_u_bottom, knot_vector_v_bottom, N, R)

# Determine k and l from knot vectors and number of control points
k = len(knot_vector_u_bottom) - (n+1) - 1
l = len(knot_vector_v_bottom) - (n+1) - 1

# Evaluate the actual function f and our approximation F over this mesh
F_values = np.zeros_like(U)
f_values = np.zeros_like(U)

for i in range(U.shape[0]):
    for j in range(U.shape[1]):
        sum_val = 0
        for p in range(n+1):
            for q in range(n+1):
                R_pq = R(p, q, k1, l1, U[i, j], V[i, j], kv_u_bottom, kv_v_bottom, weights_bottom, N)
                sum_val += beta_ij[p, q] * R_pq
        F_values[i, j] = sum_val
        f_values[i, j] = random_function_space(X[i, j], Y[i, j], Z[i, j])

# Compute the maximum error
print('The Maximum Error is: ', np.max(np.abs(F_values - f_values)))


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