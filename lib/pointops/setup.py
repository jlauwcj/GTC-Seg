from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
from distutils.sysconfig import get_config_vars
import torch  # 导入 torch
import numpy as np  # 导入 numpy 以获取头文件路径

# ----------------------
# 添加系统头文件路径（解决 crypt.h 缺失）
# ----------------------
system_includes = ['/usr/include']  # 添加系统头文件目录

# 获取 PyTorch 和 NumPy 头文件路径（保持不变）
torch_include = os.path.join(os.path.dirname(torch.__file__), "include")
numpy_include = np.get_include()

# ----------------------
# 移除严格原型警告（保持不变）
# ----------------------
(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(flag for flag in opt.split() if flag != '-Wstrict-prototypes')

setup(
    name='pointops',
    author='Hengshuang Zhao',
    ext_modules=[
        CUDAExtension(
            'pointops_cuda',
            [
                # 源文件列表（保持不变）
                'src/pointops_api.cpp',
                'src/knnquery/knnquery_cuda.cpp',
                'src/knnquery/knnquery_cuda_kernel.cu',
                'src/sampling/sampling_cuda.cpp',
                'src/sampling/sampling_cuda_kernel.cu',
                'src/grouping/grouping_cuda.cpp',
                'src/grouping/grouping_cuda_kernel.cu',
                'src/interpolation/interpolation_cuda.cpp',
                'src/interpolation/interpolation_cuda_kernel.cu',
                'src/subtraction/subtraction_cuda.cpp',
                'src/subtraction/subtraction_cuda_kernel.cu',
                'src/aggregation/aggregation_cuda.cpp',
                'src/aggregation/aggregation_cuda_kernel.cu',
            ],
            include_dirs=[
                torch_include,          # PyTorch 头文件
                numpy_include,          # NumPy 头文件
                '/usr/local/cuda/include',  # CUDA 头文件
                *system_includes,       # 新增：系统头文件（如 /usr/include）
            ],
            extra_compile_args={
                'cxx': [
                    '-O3', 
                    '-std=c++17', 
                    '-fopenmp',         # 启用 OpenMP（确保 CPU 并行支持）
                    '-D_GLIBCXX_USE_CXX11_ABI=0'  # 兼容 PyTorch ABI
                ],
                'nvcc': [
                    '-O3', 
                    '-arch=sm_86',      # 根据 GPU 架构调整（RTX 30 系列为 sm_86）
                    '--expt-relaxed-constexpr',
                    '-Xcompiler=-fopenmp',  # 向 nvcc 传递 OpenMP 标志
                    '-D__CUDA_NO_HALF_OPERATORS__',  # 避免半精度运算符警告
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)}
)
