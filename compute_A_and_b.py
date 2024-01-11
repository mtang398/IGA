import numpy as np

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