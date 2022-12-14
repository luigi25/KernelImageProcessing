cmake_minimum_required(VERSION 3.22)
project(KernelImageProcessingPThread)
message(STATUS "Checking OpenMP")
find_package(OpenMP)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE-DIRECTORIES})
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
add_executable(kernel_image_processing main.cpp PaddedImage/PaddedImage.cpp PaddedImage/PaddedImage.h PaddedImage/imageReconstruction.h KernelProcessingTests/sequentialKernelTest.h KernelProcessingTests/parallelKernelTestRowsDivision.h KernelProcessingTests/parallelKernelTestRowsColumnsDivision.h KernelProcessingTests/parallelKernelTestBlocks.h Kernel/AbstractKernel.h Kernel/GaussianKernel5x5.cpp Kernel/GaussianKernel5x5.h Kernel/GaussianKernel3x3.cpp Kernel/GaussianKernel3x3.h)
target_link_libraries(kernel_image_processing ${OpenCV_LIBS})
