#include <cuda.h>
#include "../PaddedImage/FlatPaddedImage.h"
#include "../PaddedImage/imageReconstruction.h"
#include "../Kernel/AbstractKernel.h"

//Check CUDA Errors
cudaError_t checkCudaShared(cudaError_t result){
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

const unsigned int max_kernel_size = 25;
__device__ __constant__ float gaussianKernelDevice[max_kernel_size];

__global__ void shared_kernel_convolution_3D(float* flatPaddedImage, int originalWidth, int originalHeight, int numChannels, int padding, float* flatBlurredImage, int kernelDim, float scalarValue) {
    extern __shared__ float shared_data[];

    unsigned int blockWidth = blockDim.x - (2*padding);
    unsigned int blockHeight = blockDim.y - (2*padding);
    unsigned int tileWidth = blockDim.x;
    unsigned int tileHeight = blockDim.y;

    unsigned int blockStartCol = blockIdx.x * blockWidth + padding;
    unsigned int blockEndCol = blockStartCol + blockWidth;
    unsigned int blockStartRow = blockIdx.y * blockHeight + padding;
    unsigned int blockEndRow = blockStartRow + blockHeight;

    unsigned int tileStartCol = blockStartCol - padding;
    unsigned int tileEndCol = blockEndCol + padding;

    unsigned int tileStartRow = blockStartRow - padding;
    unsigned int tileEndRow = blockEndRow + padding;


    unsigned int tilePixelPosCol = threadIdx.x;
    unsigned int pixelPosCol = tileStartCol + tilePixelPosCol;

    unsigned int tilePixelPosRow;
    unsigned int pixelPosRow;
    unsigned int pixelPos;
    unsigned int tilePixelPos;
    unsigned int maskIndex;
    unsigned int outputPixelPos;
    unsigned int blocksInTile = (int)(ceil((float)tileHeight / (float)blockDim.y));

    for (int b = 0; b < blocksInTile; b++) {
        tilePixelPosRow = threadIdx.y + b * blockDim.y;
        pixelPosRow = tileStartRow + tilePixelPosRow;
        // Check if the pixel is in the image
        if (pixelPosCol < tileEndCol && pixelPosCol < (originalWidth + 2*padding) &&
            pixelPosRow < tileEndRow && pixelPosRow < (originalHeight + 2*padding)) {
            pixelPos = (pixelPosRow * (originalWidth + 2*padding) * numChannels) + (pixelPosCol * numChannels);
            tilePixelPos = (tilePixelPosRow * tileWidth * numChannels) + (tilePixelPosCol * numChannels);
            // Load the pixel in the shared memory
            shared_data[tilePixelPos] = flatPaddedImage[pixelPos];
            shared_data[tilePixelPos + 1] = flatPaddedImage[pixelPos + 1];
            shared_data[tilePixelPos + 2] = flatPaddedImage[pixelPos + 2];
        }
    }

    __syncthreads();

    for (int b = 0; b < blocksInTile; b++) {
        tilePixelPosRow = threadIdx.y + b * blockDim.y;
        pixelPosRow = tileStartRow + tilePixelPosRow;

        if (pixelPosCol >= tileStartCol + padding && pixelPosCol < tileEndCol - padding && pixelPosCol < (originalWidth + 2*padding) &&
            pixelPosRow >= tileStartRow + padding && pixelPosRow < tileEndRow - padding && pixelPosRow < (originalHeight + 2*padding)) {

            float pixValR = 0;
            float pixValG = 0;
            float pixValB = 0;
            for (int k = -padding; k < kernelDim - padding; k++) {
                for (int l = -padding; l < kernelDim - padding; l++) {
                    tilePixelPos = ((tilePixelPosRow + k) * tileWidth * numChannels) + ((tilePixelPosCol + l) * numChannels);
                    maskIndex = (k + padding) * kernelDim + (l + padding);
                    pixValR += shared_data[tilePixelPos] * gaussianKernelDevice[maskIndex];
                    pixValG += shared_data[tilePixelPos + 1] * gaussianKernelDevice[maskIndex];
                    pixValB += shared_data[tilePixelPos + 2] * gaussianKernelDevice[maskIndex];
                }
            }

            outputPixelPos = (pixelPosRow * (originalWidth + 2*padding) * numChannels) + (pixelPosCol * numChannels);
            flatBlurredImage[outputPixelPos] = pixValR / scalarValue;
            flatBlurredImage[outputPixelPos + 1] = pixValG / scalarValue;
            flatBlurredImage[outputPixelPos + 2] = pixValB / scalarValue;
        }
    }
}

vector<vector<double>> CUDASharedKernelTest(int numExecutions, int numBlocks, const FlatPaddedImage& paddedImage, AbstractKernel& kernel) {
    vector<double> meanExecutionsTimeVec;
    vector<double> meanCopyTimeVec;
    for (int blockDimension = 2; blockDimension <= numBlocks; blockDimension *= 2) {
        if (blockDimension > 2 * paddedImage.getPadding()) {
            double meanExecutionsTime = 0;
            double meanCopyTime = 0;
            for (int execution = 0; execution < numExecutions; execution++) {
                float *flatPaddedImage;
                float *flatPaddedImageDevice;

                int originalWidth = paddedImage.getOriginalWidth();
                int originalHeight = paddedImage.getOriginalHeight();
                int numChannels = paddedImage.getNumChannels();
                int padding = paddedImage.getPadding();
                int paddedSize = (originalWidth + (padding * 2)) * (originalHeight + (padding * 2)) * numChannels;
                float *flatBlurredImage;
                float *flatBlurredImageDevice;

                float gaussianKernel[max_kernel_size];
                int kernelDim = kernel.getKernelDimension();
                int kernelSize = kernel.getKernelSize();
                float scalarValue = kernel.getScalarValue();

                const int tileWidth = blockDimension;
                const int tileHeight = blockDimension;

                const int blockWidth = tileWidth - 2 * padding;
                const int blockHeight = tileHeight - 2 * padding;

                checkCudaShared(cudaMallocHost((void **) &flatPaddedImage, sizeof(float) * paddedSize));
                checkCudaShared(cudaMallocHost((void **) &flatBlurredImage, sizeof(float) * paddedSize));
                checkCudaShared(cudaMallocHost((void **) &gaussianKernel, sizeof(float) * max_kernel_size));

                flatPaddedImage = paddedImage.getFlatPaddedImage();
                float *gaussianKernel_temp = kernel.getFlatKernel();

                for (int i = 0; i < max_kernel_size; i++) {
                    if (i < kernelSize)
                        gaussianKernel[i] = gaussianKernel_temp[i];
                    else
                        gaussianKernel[i] = 0;
                }

                //allocate device memory
                auto startCopy = chrono::system_clock::now();
                checkCudaShared(cudaMalloc((void **) &flatPaddedImageDevice, sizeof(float) * paddedSize));
                checkCudaShared(cudaMalloc((void **) &flatBlurredImageDevice, sizeof(float) * paddedSize));

                //transfer data from host to device memory
                checkCudaShared(cudaMemcpy(flatPaddedImageDevice, flatPaddedImage, sizeof(float) * paddedSize,
                                           cudaMemcpyHostToDevice));
                checkCudaShared(cudaMemcpy(flatBlurredImageDevice, flatBlurredImage, sizeof(float) * paddedSize,
                                           cudaMemcpyHostToDevice));
                checkCudaShared(
                        cudaMemcpyToSymbol(gaussianKernelDevice, gaussianKernel, sizeof(float) * max_kernel_size, 0,
                                           cudaMemcpyHostToDevice));
                chrono::duration<double> endCopy{};
                endCopy = chrono::system_clock::now() - startCopy;
                auto copyTime = chrono::duration_cast<chrono::microseconds>(endCopy);
                meanCopyTime += (double) copyTime.count();

                dim3 DimGrid((int) ceil((float) (originalWidth + (padding * 2)) / (float) blockWidth),
                             (int) ceil((float) (originalHeight + (padding * 2)) / (float) blockHeight));
                dim3 DimBlock(tileWidth, tileHeight);
                int sharedMemorySize = tileWidth * tileHeight * numChannels * sizeof(float);

                auto start = std::chrono::system_clock::now();
                shared_kernel_convolution_3D<<<DimGrid, DimBlock, sharedMemorySize>>>(flatPaddedImageDevice,
                                                                                      originalWidth, originalHeight,
                                                                                      numChannels, padding,
                                                                                      flatBlurredImageDevice, kernelDim,
                                                                                      scalarValue);
                cudaDeviceSynchronize();

                chrono::duration<double> executionTime{};
                executionTime = chrono::system_clock::now() - start;
                auto executionTimeMicroseconds = chrono::duration_cast<chrono::microseconds>(executionTime);
                meanExecutionsTime += (double) executionTimeMicroseconds.count();

                // transfer data back to host memory
                startCopy = chrono::system_clock::now();
                checkCudaShared(cudaMemcpy(flatBlurredImage, flatBlurredImageDevice, sizeof(float) * paddedSize,
                                           cudaMemcpyDeviceToHost));
                endCopy = chrono::system_clock::now() - startCopy;
                copyTime = chrono::duration_cast<chrono::microseconds>(endCopy);
                meanCopyTime += (double) copyTime.count();

//                Mat reconstructed_image = imageReconstruction(flatBlurredImage, originalWidth, originalHeight, numChannels, padding);
//                imwrite("../results/blurred_" + to_string(blockDimension) + ".jpeg", reconstructed_image);

                // deallocate device memory
                checkCudaShared(cudaFree(flatPaddedImageDevice));
                checkCudaShared(cudaFree(flatBlurredImageDevice));

                cudaFreeHost(flatPaddedImage);
                cudaFreeHost(flatBlurredImage);
                cudaFreeHost(gaussianKernel);
                cudaDeviceReset();
            }
            meanExecutionsTimeVec.push_back(meanExecutionsTime / numExecutions);
            meanCopyTimeVec.push_back(meanCopyTime / numExecutions);
        }
    }
    vector<vector<double>> execTimeVec;
    execTimeVec.push_back(meanExecutionsTimeVec);
    execTimeVec.push_back(meanCopyTimeVec);
    return execTimeVec;
}