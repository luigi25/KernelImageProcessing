#include "GaussianKernel5x5.h"

GaussianKernel5x5::GaussianKernel5x5() {
    kernelDimension = 5;
    kernel = new int*[kernelDimension];
    kernel[0] = new int[kernelDimension] {1, 4, 6, 4, 1};
    kernel[1] = new int[kernelDimension] {4, 16, 24, 16, 4};
    kernel[2] = new int[kernelDimension] {6, 24, 36, 24, 6};
    kernel[3] = new int[kernelDimension] {4, 16, 24, 16, 4};
    kernel[4] = new int[kernelDimension] {1, 4, 6, 4, 1};
    kernelSize = 25;
    scalarValue = 256;
    padding = 2;
}

int GaussianKernel5x5::getPadding() {
    return padding;
}

int** GaussianKernel5x5::getKernel() {
    return kernel;
}

int GaussianKernel5x5::getKernelSize() {
    return kernelSize;
}

int GaussianKernel5x5::getScalarValue() {
    return scalarValue;
}

int GaussianKernel5x5::getKernelDimension() {
    return kernelDimension;
}

