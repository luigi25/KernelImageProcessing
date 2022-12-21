#ifndef KERNELIMAGEPROCESSINGCUDA_SEQUENTIALKERNELTEST_H
#define KERNELIMAGEPROCESSINGCUDA_SEQUENTIALKERNELTEST_H
#include "../PaddedImage/FlatPaddedImage.h"
#include "../PaddedImage/imageReconstruction.h"
#include "../Kernel/AbstractKernel.h"

void sequential_convolution_3D(float* flatPaddedImage, int originalWidth, int originalHeight, int numChannels, int padding, float* flatBlurredImage, float* gaussianKernel, int kernelDim, float scalarValue) {
    for (int i = 0; i < (originalHeight + 2*padding); i++){
        for (int j = 0; j < (originalWidth + 2 * padding); j++){
            if (j > 1 && j < (originalWidth + padding) && i > 1 && i < (originalHeight + padding)) {
                float pixValR = 0;
                float pixValG = 0;
                float pixValB = 0;
                // Get the of the surrounding box
                for(int k = -padding; k < kernelDim - padding; k++) {
                    for(int l = -padding; l < kernelDim - padding; l++) {
                        pixValR += flatPaddedImage[((i + k) * (originalWidth + 2*padding) * numChannels) + ((j + l) * numChannels)] * gaussianKernel[((l + padding) * kernelDim) + (l + padding)];
                        pixValG += flatPaddedImage[((i + k) * (originalWidth + 2*padding) * numChannels) + ((j + l) * numChannels) + 1] * gaussianKernel[((l + padding) * kernelDim) + (l + padding)];
                        pixValB += flatPaddedImage[((i + k) * (originalWidth + 2*padding) * numChannels) + ((j + l) * numChannels) + 2] * gaussianKernel[((l + padding) * kernelDim) + (l + padding)];

                    }
                }
                // Write our new pixel value out
                flatBlurredImage[(i * (originalWidth + 2*padding) * numChannels) + (j * numChannels)] = pixValR / scalarValue;
                flatBlurredImage[(i * (originalWidth + 2*padding) * numChannels) + (j * numChannels) + 1] = pixValG / scalarValue;
                flatBlurredImage[(i * (originalWidth + 2*padding) * numChannels) + (j * numChannels) + 2] = pixValB / scalarValue;
            }
        }
    }
}

double sequentialTest(int numExecutions, const FlatPaddedImage& paddedImage, AbstractKernel& kernel) {
    double meanExecutionsTime = 0;
    float* flatPaddedImage = paddedImage.getFlatPaddedImage();
    int originalWidth = paddedImage.getOriginalWidth();
    int originalHeight = paddedImage.getOriginalHeight();
    int numChannels = paddedImage.getNumChannels();
    int padding = paddedImage.getPadding();
    int paddedSize = (originalWidth+(padding*2)) * (originalHeight+(padding*2)) * numChannels;
    float* gaussianKernel = kernel.getFlatKernel();
    int kernelDim = kernel.getKernelDimension();
    float scalarValue = kernel.getScalarValue();
    for (int execution = 0; execution < numExecutions; execution++) {
        float* flatBlurredImage = new float[paddedSize];
        auto start = std::chrono::system_clock::now();
        sequential_convolution_3D(flatPaddedImage, originalWidth, originalHeight, numChannels, padding, flatBlurredImage,
                                  gaussianKernel, kernelDim, scalarValue);
        chrono::duration<double> executionTime{};
        executionTime = chrono::system_clock::now() - start;
        auto executionTimeMicroseconds = chrono::duration_cast<chrono::microseconds>(executionTime);
        meanExecutionsTime += (double)executionTimeMicroseconds.count();
//        Mat reconstructed_image = imageReconstruction(flatBlurredImage, originalWidth, originalHeight, numChannels, padding);
//        imwrite("../results/prova.jpg", reconstructed_image);
        free(flatBlurredImage);
    }
    return meanExecutionsTime / numExecutions;


}
#endif //KERNELIMAGEPROCESSINGCUDA_SEQUENTIALKERNELTEST_H
