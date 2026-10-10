from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(
    name="inference",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="inference._C",
            sources=[
                "bindings.cpp",
                "add.cu",
                "mul.cu",
                "gemm.cu",
                "rmsnorm.cu",
                "softmax.cu",
                "rope.cu"
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-arch=sm_89", "-Xptxas=-v"]
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)