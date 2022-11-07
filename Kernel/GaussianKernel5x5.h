#ifndef KERNELIMAGEPROCESSING_GAUSSIANKERNEL5X5_H
#define KERNELIMAGEPROCESSING_GAUSSIANKERNEL5X5_H

#include "AbstractKernel.h"

class GaussianKernel5x5 : public AbstractKernel {
public:
    explicit GaussianKernel5x5();
    int getPadding() override;
    float** getKernel() override;
    int getKernelSize() override;
    float getScalarValue() override;
    int getKernelDimension() override;
};


#endif //KERNELIMAGEPROCESSING_GAUSSIANKERNEL5X5_H
