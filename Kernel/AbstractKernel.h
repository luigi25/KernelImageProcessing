#ifndef KERNELIMAGEPROCESSING_ABSTRACTKERNEL_H
#define KERNELIMAGEPROCESSING_ABSTRACTKERNEL_H

#include <vector>

using namespace std;

class AbstractKernel {
protected:
    int kernelDimension = 0;
    int kernelSize = 0;
    int scalarValue = 0;
    int padding = 0;
    int** kernel{};

public:
    virtual int** getKernel() = 0;
    virtual int getKernelSize() = 0;
    virtual int getScalarValue() = 0;
    virtual int getPadding() = 0;
    virtual int getKernelDimension() = 0;
};


#endif //KERNELIMAGEPROCESSING_ABSTRACTKERNEL_H
