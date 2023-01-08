cmake_minimum_required(VERSION 3.22)
project(KernelImageProcessingCUDA CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(kernel_image_processing main.cu PaddedImage/FlatPaddedImage.cpp PaddedImage/FlatPaddedImage.h PaddedImage/imageReconstruction.h Kernel/AbstractKernel.h Kernel/GaussianKernel5x5.cpp Kernel/GaussianKernel5x5.h Kernel/GaussianKernel3x3.cpp Kernel/GaussianKernel3x3.h KernelProcessingTests/SequentialKernelTest.h KernelProcessingTests/CUDAKernelTestGlobal.h KernelProcessingTests/CUDAKernelTestConstant.h KernelProcessingTests/CUDAKernelTestShared.h)
target_link_libraries(kernel_image_processing ${OpenCV_LIBS})
set_target_properties(kernel_image_processing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

