from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    name = "bilateral",
    version = "0.9",
    description = "2D bilateral filter",
    author = "Nadav Horesh",
    author_email = "nadavh at visionsense dot com",
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("bilateral_base", ["bilateral_base.pyx"],
                             include_dirs = [numpy.get_include()],
                             extra_compile_args=['-O3'])
                   ],
    ext_package = "bilateral",
    py_modules = ["bilateral/bilateral", "bilateral/__init__"]
)
