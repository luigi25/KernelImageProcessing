#include "GaussianKernel5x5.h"

GaussianKernel5x5::GaussianKernel5x5() {
    kernelDimension = 5;
    kernel = new float*[kernelDimension];
    kernel[0] = new float[kernelDimension] {1, 4, 6, 4, 1};
    kernel[1] = new float[kernelDimension] {4, 16, 24, 16, 4};
    kernel[2] = new float[kernelDimension] {6, 24, 36, 24, 6};
    kernel[3] = new float[kernelDimension] {4, 16, 24, 16, 4};
    kernel[4] = new float[kernelDimension] {1, 4, 6, 4, 1};
    kernelSize = 25;
    scalarValue = 256;
}

float** GaussianKernel5x5::getKernel() {
    return kernel;
}

int GaussianKernel5x5::getKernelSize() {
    return kernelSize;
}

float GaussianKernel5x5::getScalarValue() {
    return scalarValue;
}

int GaussianKernel5x5::getKernelDimension() {
    return kernelDimension;
}

