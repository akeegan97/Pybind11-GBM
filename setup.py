from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "gbmapp.simulation",
        ["src/cpp/simulation.cpp"],
        cxx_std=17,
        extra_compile_args=[
            "-O3",
            "-mavx",
            "-mavx2",
            "-mfma",
        ],
    ),
]
setup(
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
