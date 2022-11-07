#include "GaussianKernel3x3.h"

GaussianKernel3x3::GaussianKernel3x3() {
    kernelDimension = 3;
    kernel = new float*[kernelDimension];
    kernel[0] = new float[kernelDimension] {1, 2, 1};
    kernel[1] = new float[kernelDimension] {2, 4, 2};
    kernel[2] = new float[kernelDimension] {1, 2, 1};
    kernelSize = 9;
    scalarValue = 16;
    padding = 1;
}

int GaussianKernel3x3::getPadding() {
    return padding;
}

float** GaussianKernel3x3::getKernel() {
    return kernel;
}

int GaussianKernel3x3::getKernelSize() {
    return kernelSize;
}

float GaussianKernel3x3::getScalarValue() {
    return scalarValue;
}

int GaussianKernel3x3::getKernelDimension() {
    return kernelDimension;
}
