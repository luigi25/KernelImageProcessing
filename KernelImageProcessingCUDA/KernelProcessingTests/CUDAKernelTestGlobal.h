#include <cuda.h>
#include "../PaddedImage/FlatPaddedImage.h"
#include "../PaddedImage/imageReconstruction.h"
#include "../Kernel/AbstractKernel.h"

//Check CUDA Errors
cudaError_t checkCudaGlobal(cudaError_t result){
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__ void global_kernel_convolution_3D(float* flatPaddedImage, int originalWidth, int originalHeight, int numChannels, int padding, float* flatBlurredImage, float* gaussianKernel, int kernelDim, float scalarValue) {
    // set indices
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int maskIndex;
    unsigned int pixelPos;
    unsigned int outputPixelPos;

    // check if the position is in the original image and not in padding
    if (x >= padding && x < (originalWidth + padding) && y >= padding && y < (originalHeight + padding)) {
        // apply filtering
        float pixValR = 0;
        float pixValG = 0;
        float pixValB = 0;
        for(int k = -padding; k < kernelDim - padding; k++) {
            for(int l = -padding; l < kernelDim - padding; l++) {
                pixelPos = ((y + k) * (originalWidth + 2*padding) * numChannels) + ((x + l) * numChannels);
                maskIndex = (k + padding) * kernelDim + (l + padding);
                pixValR += flatPaddedImage[pixelPos] * gaussianKernel[maskIndex];
                pixValG += flatPaddedImage[pixelPos + 1] * gaussianKernel[maskIndex];
                pixValB += flatPaddedImage[pixelPos + 2] * gaussianKernel[maskIndex];
            }
        }
        // write new pixel value in output image
        outputPixelPos = (y * (originalWidth + 2*padding) * numChannels) + (x * numChannels);
        flatBlurredImage[outputPixelPos] = pixValR / scalarValue;
        flatBlurredImage[outputPixelPos + 1] = pixValG / scalarValue;
        flatBlurredImage[outputPixelPos + 2] = pixValB / scalarValue;
    }
}

vector<double> CUDAGlobalKernelTest(int numExecutions, int numBlocks, const FlatPaddedImage& paddedImage, AbstractKernel& kernel) {
    vector<double> meanExecutionsTimeVec;
    for (int blockDimension = 2; blockDimension <= numBlocks; blockDimension *= 2) {
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

            // allocate host memory
            checkCudaGlobal(cudaMallocHost((void **) &flatPaddedImage, sizeof(float) * paddedSize));
            checkCudaGlobal(cudaMallocHost((void **) &flatBlurredImage, sizeof(float) * paddedSize));
            checkCudaGlobal(cudaMallocHost((void **) &gaussianKernel, sizeof(float) * kernelSize));

            flatPaddedImage = paddedImage.getFlatPaddedImage();
            gaussianKernel = kernel.getFlatKernel();

            // allocate device memory
            checkCudaGlobal(cudaMalloc((void **) &flatPaddedImage_device, sizeof(float) * paddedSize));
            checkCudaGlobal(cudaMalloc((void **) &flatBlurredImage_device, sizeof(float) * paddedSize));
            checkCudaGlobal(cudaMalloc((void **) &gaussianKernel_device, sizeof(float) * kernelSize));

            // transfer data from host to device memory
            checkCudaGlobal(cudaMemcpy(flatPaddedImage_device, flatPaddedImage, sizeof(float) * paddedSize,
                                 cudaMemcpyHostToDevice));
            checkCudaGlobal(cudaMemcpy(flatBlurredImage_device, flatBlurredImage, sizeof(float) * paddedSize,
                                 cudaMemcpyHostToDevice));
            checkCudaGlobal(
                    cudaMemcpy(gaussianKernel_device, gaussianKernel, sizeof(float) * kernelSize, cudaMemcpyHostToDevice));

            // define DimGrid and DimBlock
            dim3 DimGrid((int) ceil((float) (originalWidth + (padding * 2)) / (float) blockDimension),
                         (int) ceil((float) (originalHeight + (padding * 2)) / (float) blockDimension));
            dim3 DimBlock(blockDimension, blockDimension);

            auto start = std::chrono::system_clock::now();
            // start global convolution
            global_kernel_convolution_3D<<<DimGrid, DimBlock>>>(flatPaddedImage_device, originalWidth, originalHeight,
                                                                  numChannels,
                                                                  padding, flatBlurredImage_device,
                                                                  gaussianKernel_device,
                                                                  kernelDim,
                                                                  scalarValue);
            cudaDeviceSynchronize();

            chrono::duration<double> executionTime{};
            executionTime = chrono::system_clock::now() - start;
            auto executionTimeMicroseconds = chrono::duration_cast<chrono::microseconds>(executionTime);
            meanExecutionsTime += (double)executionTimeMicroseconds.count();

            // transfer data back to host memory
            checkCudaGlobal(cudaMemcpy(flatBlurredImage, flatBlurredImage_device, sizeof(float) * paddedSize, cudaMemcpyDeviceToHost));

//            Mat reconstructed_image = imageReconstruction(flatBlurredImage, originalWidth, originalHeight, numChannels, padding);
//            imwrite("../results/blurred_" + to_string(blockDimension) + ".jpeg", reconstructed_image);

            // deallocate device memory
            checkCudaGlobal(cudaFree(flatPaddedImage_device));
            checkCudaGlobal(cudaFree(flatBlurredImage_device));
            checkCudaGlobal(cudaFree(gaussianKernel_device));

            // free host memory
            cudaFreeHost(flatPaddedImage);
            cudaFreeHost(flatBlurredImage);
            cudaFreeHost(gaussianKernel);
            cudaDeviceReset();
        }
        meanExecutionsTimeVec.push_back(meanExecutionsTime / numExecutions);
    }
    return meanExecutionsTimeVec;
}