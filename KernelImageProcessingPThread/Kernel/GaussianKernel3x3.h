#ifndef KERNELIMAGEPROCESSING_GAUSSIANKERNEL3X3_H
#define KERNELIMAGEPROCESSING_GAUSSIANKERNEL3X3_H

#include "AbstractKernel.h"

class GaussianKernel3x3 : public AbstractKernel {
public:
    explicit GaussianKernel3x3();
    float** getKernel() override;
    int getKernelSize() override;
    float getScalarValue() override;
    int getKernelDimension() override;
};


#endif //KERNELIMAGEPROCESSING_GAUSSIANKERNEL3X3_H
