# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

header_directory = '.'  # Substitute with the actual directory if it's not in the current directory

# Define the extension
ext = Extension(
    name="mutual_info",
    sources=["mutual_info.pyx"],
    language="c++",
    extra_compile_args=["-O3", "-funroll-loops", "-march=native"],  # Use -O3 for optimization
    include_dirs=[numpy.get_include(), header_directory],  # Include the header file directory
    extra_link_args=[],
)

# Setup the module
setup(
    name="mutual_info",
    ext_modules=cythonize([ext]),
    zip_safe=False,
)