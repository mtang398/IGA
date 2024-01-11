from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("compute_A_and_b_spline.pyx"),
    include_dirs=[np.get_include()]
)
