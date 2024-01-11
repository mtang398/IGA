from cffi import FFI

ffibuilder = FFI()

# Read the C source file
with open("nurbs_functions.c", "r") as f:
    c_source = f.read()

ffibuilder.set_source("nurbs_c", c_source)

# Define the function signatures and types we want to make accessible from Python
ffibuilder.cdef("""
    typedef struct {
        int i;
        int k;
        double t;
        double value;
    } CacheItem;

    double N(int i, int k, double t, double knot_vector[], int knot_vector_size, CacheItem cache[], int *cache_count);
    double R(int i, int j, int k, int l, double u, double v, double *knot_vector_u, int knot_vector_u_size, double *knot_vector_v, int knot_vector_v_size, double *weights, int weights_size, CacheItem *cache_u, int *cache_count_u, CacheItem *cache_v, int *cache_count_v);
    void surface_point(double u, double v, double *control_points, int control_points_rows, int control_points_cols, double *weights, int weights_size, double *knot_vector_u, int knot_vector_u_size, double *knot_vector_v, int knot_vector_v_size, double point[3]);
    void bernstein_basis(int i, int n, double *u_array, double *result, int array_size);
    void tensor_bernstein_basis(int i, int j, int n, double *u_array, double *v_array, double **result, int u_size, int v_size);

""")

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)

