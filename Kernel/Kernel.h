#ifndef KERNELIMAGEPROCESSING_KERNEL_H
#define KERNELIMAGEPROCESSING_KERNEL_H

#include <vector>

using namespace std;

class Kernel {
private:
    int kernelSize;
    int scalarValue;
    int** kernel;

public:
    explicit Kernel(int** _kernel, int _kernelSize, int _scalarValue);
    int** getKernel();
    int getKernelSize() const;
    int getScalarValue() const;
};


#endif //KERNELIMAGEPROCESSING_KERNEL_H
