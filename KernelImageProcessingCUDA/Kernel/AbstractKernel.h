#ifndef KERNELIMAGEPROCESSING_ABSTRACTKERNEL_H
#define KERNELIMAGEPROCESSING_ABSTRACTKERNEL_H

#include <vector>

using namespace std;

class AbstractKernel {
protected:
    int kernelDimension = 0;
    int kernelSize = 0;
    float scalarValue = 0;
    float** kernel{};

public:
    virtual float** getKernel() = 0;
    virtual float* getFlatKernel() = 0;
    virtual int getKernelSize() = 0;
    virtual float getScalarValue() = 0;
    virtual int getKernelDimension() = 0;
};


#endif //KERNELIMAGEPROCESSING_ABSTRACTKERNEL_H
