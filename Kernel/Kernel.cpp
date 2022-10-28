#include "Kernel.h"

Kernel::Kernel(int** _kernel, int _kernelSize, int _scalarValue) {
    kernel = _kernel;
    kernelSize = _kernelSize;
    scalarValue = _scalarValue;
}

int** Kernel::getKernel(){
    return kernel;
}

int Kernel::getKernelSize() const {
    return kernelSize;
}

int Kernel::getScalarValue() const {
    return scalarValue;
}