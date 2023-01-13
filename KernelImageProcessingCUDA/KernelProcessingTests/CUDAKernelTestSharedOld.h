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

// define gaussianKernel in constant memory
#define MASK_WIDTH 5
#define BLOCK_WIDTH 32

__device__ __constant__ float gaussianKernelDevice[MASK_WIDTH * MASK_WIDTH];

__global__ void shared_kernel_convolution_3D(float* flatPaddedImage, int originalWidth, int originalHeight, int numChannels, int padding, float* flatBlurredImage, int kernelDim, float scalarValue) {
    // 3 = numChannels
    const int size = BLOCK_WIDTH * BLOCK_WIDTH * 3;
    __shared__ float shared_data[size];

    // define width, height for block and tile
    unsigned int blockWidth = blockDim.x - (2*padding);
    unsigned int blockHeight = blockDim.y - (2*padding);
    unsigned int tileWidth = blockDim.x;

    // define rows, columns start/end indices for block and tile
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

    // load pixel values in shared memory
    tilePixelPosRow = threadIdx.y;
    pixelPosRow = tileStartRow + tilePixelPosRow;
    // check if the pixel is in the image
    if (pixelPosCol < tileEndCol && pixelPosCol < (originalWidth + 2*padding) &&
        pixelPosRow < tileEndRow && pixelPosRow < (originalHeight + 2*padding)) {
        pixelPos = (pixelPosRow * (originalWidth + 2*padding) * numChannels) + (pixelPosCol * numChannels);
        tilePixelPos = (tilePixelPosRow * tileWidth * numChannels) + (tilePixelPosCol * numChannels);

        // load the pixel in the shared memory
        shared_data[tilePixelPos] = flatPaddedImage[pixelPos];
        shared_data[tilePixelPos + 1] = flatPaddedImage[pixelPos + 1];
        shared_data[tilePixelPos + 2] = flatPaddedImage[pixelPos + 2];
    }

    __syncthreads();

    // apply filtering
    // check if the position is in the original image and not in padding
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
        // write new pixel value in output image
        outputPixelPos = (pixelPosRow * (originalWidth + 2*padding) * numChannels) + (pixelPosCol * numChannels);
        flatBlurredImage[outputPixelPos] = pixValR / scalarValue;
        flatBlurredImage[outputPixelPos + 1] = pixValG / scalarValue;
        flatBlurredImage[outputPixelPos + 2] = pixValB / scalarValue;
    }
}

vector<double> CUDASharedKernelTest(int numExecutions, int numBlocks, const FlatPaddedImage& paddedImage, AbstractKernel& kernel) {
    vector<double> meanExecutionsTimeVec;
    double meanExecutionsTime = 0;
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

        float gaussianKernel[MASK_WIDTH * MASK_WIDTH];
        int kernelDim = kernel.getKernelDimension();
        int kernelSize = kernel.getKernelSize();
        float scalarValue = kernel.getScalarValue();

        const int tileWidth = BLOCK_WIDTH;
        const int tileHeight = BLOCK_WIDTH;

        const int blockWidth = tileWidth - 2 * padding;
        const int blockHeight = tileHeight - 2 * padding;

        // allocate host memory
        checkCudaShared(cudaMallocHost((void **) &flatPaddedImage, sizeof(float) * paddedSize));
        checkCudaShared(cudaMallocHost((void **) &flatBlurredImage, sizeof(float) * paddedSize));
        checkCudaShared(cudaMallocHost((void **) &gaussianKernel, sizeof(float) * MASK_WIDTH * MASK_WIDTH));

        flatPaddedImage = paddedImage.getFlatPaddedImage();
        float *gaussianKernel_temp = kernel.getFlatKernel();

        // initialize gaussianKernel
        for (int i = 0; i < MASK_WIDTH * MASK_WIDTH; i++) {
            gaussianKernel[i] = gaussianKernel_temp[i];
        }

        // allocate device memory
        checkCudaShared(cudaMalloc((void **) &flatPaddedImageDevice, sizeof(float) * paddedSize));
        checkCudaShared(cudaMalloc((void **) &flatBlurredImageDevice, sizeof(float) * paddedSize));

        // transfer data from host to device memory
        checkCudaShared(cudaMemcpy(flatPaddedImageDevice, flatPaddedImage, sizeof(float) * paddedSize,
                                   cudaMemcpyHostToDevice));
        checkCudaShared(cudaMemcpy(flatBlurredImageDevice, flatBlurredImage, sizeof(float) * paddedSize,
                                   cudaMemcpyHostToDevice));
        checkCudaShared(
                cudaMemcpyToSymbol(gaussianKernelDevice, gaussianKernel, sizeof(float) * MASK_WIDTH * MASK_WIDTH, 0,
                                   cudaMemcpyHostToDevice));

        // define DimGrid and DimBlock
        dim3 DimGrid((int) ceil((float) (originalWidth + (padding * 2)) / (float) blockWidth),
                     (int) ceil((float) (originalHeight + (padding * 2)) / (float) blockHeight));
        dim3 DimBlock(tileWidth, tileHeight);

        auto start = std::chrono::system_clock::now();
        // start shared convolution
        shared_kernel_convolution_3D<<<DimGrid, DimBlock>>>(flatPaddedImageDevice,
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
        checkCudaShared(cudaMemcpy(flatBlurredImage, flatBlurredImageDevice, sizeof(float) * paddedSize,
                                   cudaMemcpyDeviceToHost));

//        Mat reconstructed_image = imageReconstruction(flatBlurredImage, originalWidth, originalHeight, numChannels, padding);
//        imwrite("../results/blurred_" + to_string(BLOCK_WIDTH) + ".jpeg", reconstructed_image);

        // deallocate device memory
        checkCudaShared(cudaFree(flatPaddedImageDevice));
        checkCudaShared(cudaFree(flatBlurredImageDevice));

        // free host memory
        cudaFreeHost(flatPaddedImage);
        cudaFreeHost(flatBlurredImage);
        cudaFreeHost(gaussianKernel);
        cudaDeviceReset();
    }
    meanExecutionsTimeVec.push_back(meanExecutionsTime / numExecutions);
    return meanExecutionsTimeVec;
}