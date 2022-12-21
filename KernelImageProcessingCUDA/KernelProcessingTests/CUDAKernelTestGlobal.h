#include <cuda.h>
#include "../PaddedImage/FlatPaddedImage.h"
#include "../PaddedImage/imageReconstruction.h"
#include "../Kernel/AbstractKernel.h"

//Check CUDA Errors
cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__ void sequential_convolution_3D(float* flatPaddedImage, int originalWidth, int originalHeight, int numChannels, int padding, float* flatBlurredImage, float* gaussianKernel, int kernelDim, float scalarValue) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
//    printf("colX: %d\n", colX);
//    printf("rowY: %d\n", rowY);
    if (x > 1 && x < (originalWidth + padding) && y > 1 && y < (originalHeight + padding)) {
        float pixValR = 0;
        float pixValG = 0;
        float pixValB = 0;
        // Get the of the surrounding box
        for(int k = -padding; k < kernelDim - padding; k++) {
            for(int l = -padding; l < kernelDim - padding; l++) {
                pixValR += flatPaddedImage[((y + k) * (originalWidth + 2*padding) * numChannels) + ((x + l) * numChannels)] * gaussianKernel[((k + padding) * kernelDim) + (l + padding)];
                pixValG += flatPaddedImage[((y + k) * (originalWidth + 2*padding) * numChannels) + ((x + l) * numChannels) + 1] * gaussianKernel[((k + padding) * kernelDim) + (l + padding)];
                pixValB += flatPaddedImage[((y + k) * (originalWidth + 2*padding) * numChannels) + ((x + l) * numChannels) + 2] * gaussianKernel[((k + padding) * kernelDim) + (l + padding)];
            }
        }
        // Write our new pixel value out
        flatBlurredImage[(y * (originalWidth + 2*padding) * numChannels) + (x * numChannels)] = pixValR / scalarValue;
        flatBlurredImage[(y * (originalWidth + 2*padding) * numChannels) + (x * numChannels) + 1] = pixValG / scalarValue;
        flatBlurredImage[(y * (originalWidth + 2*padding) * numChannels) + (x * numChannels) + 2] = pixValB / scalarValue;
    }
//    printf("Current Position: %d\n", (y * (originalWidth + 2*padding) * numChannels) + (x * numChannels));
}

vector<double> parallelCudaTest(int numExecutions, const FlatPaddedImage& paddedImage, AbstractKernel& kernel) {
    vector<double> meanExecutionsTimeVec;
    for (int blockDimension = 2; blockDimension <= 32; blockDimension *= 2) {
        double meanExecutionsTime = 0;
        for (int execution = 0; execution < numExecutions; execution++) {
            float *flatPaddedImage;
            float *flatPaddedImage_device;

            int originalWidth = paddedImage.getOriginalWidth();
            int originalHeight = paddedImage.getOriginalHeight();
            int numChannels = paddedImage.getNumChannels();
            int padding = paddedImage.getPadding();
            int paddedSize = (originalWidth + (padding * 2)) * (originalHeight + (padding * 2)) * numChannels;
            float *flatBlurredImage;
            float *flatBlurredImage_device;

            float *gaussianKernel;
            float *gaussianKernel_device;
            int kernelDim = kernel.getKernelDimension();
            int kernelSize = kernel.getKernelSize();
            float scalarValue = kernel.getScalarValue();

            checkCuda(cudaMallocHost((void **) &flatPaddedImage, sizeof(float) * paddedSize));
            checkCuda(cudaMallocHost((void **) &flatBlurredImage, sizeof(float) * paddedSize));
            checkCuda(cudaMallocHost((void **) &gaussianKernel, sizeof(float) * kernelSize));

            flatPaddedImage = paddedImage.getFlatPaddedImage();
            gaussianKernel = kernel.getFlatKernel();

            //allocate device memory
            checkCuda(cudaMalloc((void **) &flatPaddedImage_device, sizeof(float) * paddedSize));
            checkCuda(cudaMalloc((void **) &flatBlurredImage_device, sizeof(float) * paddedSize));
            checkCuda(cudaMalloc((void **) &gaussianKernel_device, sizeof(float) * kernelSize));

            //transfer data from host to device memory
            checkCuda(cudaMemcpy(flatPaddedImage_device, flatPaddedImage, sizeof(float) * paddedSize,
                                 cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(flatBlurredImage_device, flatBlurredImage, sizeof(float) * paddedSize,
                                 cudaMemcpyHostToDevice));
            checkCuda(
                    cudaMemcpy(gaussianKernel_device, gaussianKernel, sizeof(float) * kernelSize, cudaMemcpyHostToDevice));

            dim3 DimGrid((int) ceil((float) (originalWidth + (padding * 2)) / (float) blockDimension),
                         (int) ceil((float) (originalHeight + (padding * 2)) / (float) blockDimension), 1);
            dim3 DimBlock(blockDimension, blockDimension, 1);

            auto start = std::chrono::system_clock::now();
            sequential_convolution_3D<<<DimGrid, DimBlock>>>(flatPaddedImage_device, originalWidth, originalHeight,
                                                             numChannels,
                                                             padding, flatBlurredImage_device, gaussianKernel_device,
                                                             kernelDim,
                                                             scalarValue);
            cudaDeviceSynchronize();

            chrono::duration<double> executionTime{};
            executionTime = chrono::system_clock::now() - start;
            auto executionTimeMicroseconds = chrono::duration_cast<chrono::microseconds>(executionTime);
            meanExecutionsTime += (double)executionTimeMicroseconds.count();

            // transfer data back to host memory
            checkCuda(cudaMemcpy(flatBlurredImage, flatBlurredImage_device, sizeof(float) * paddedSize, cudaMemcpyDeviceToHost));

//            Mat reconstructed_image = imageReconstruction(flatBlurredImage, originalWidth, originalHeight, numChannels, padding);
//            imwrite("../results/blurred_" + to_string(blockDimension) + ".jpeg", reconstructed_image);

            // Deallocate device memory
            checkCuda(cudaFree(flatPaddedImage_device));
            checkCuda(cudaFree(flatBlurredImage_device));
            checkCuda(cudaFree(gaussianKernel_device));

            cudaFreeHost(flatPaddedImage);
            cudaFreeHost(flatBlurredImage);
            cudaFreeHost(gaussianKernel);
        }
        meanExecutionsTimeVec.push_back(meanExecutionsTime / numExecutions);
    }
    return meanExecutionsTimeVec;
}