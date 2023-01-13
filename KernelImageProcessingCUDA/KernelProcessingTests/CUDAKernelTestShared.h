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

#define MASK_WIDTH 5
#define BLOCK_WIDTH 32

// define gaussianKernel in constant memory
const unsigned int max_kernel_size = MASK_WIDTH * MASK_WIDTH;
__device__ __constant__ float gaussianKernelDevice[max_kernel_size];

__global__ void shared_kernel_convolution_3D(float* flatPaddedImage, int paddedWidth, int paddedHeight, int numChannels, int padding, float* flatBlurredImage, int kernelDim, float scalarValue) {
    const int size = BLOCK_WIDTH + MASK_WIDTH - 1;
    __shared__ float dataR_ds[size][size];
    __shared__ float dataG_ds[size][size];
    __shared__ float dataB_ds[size][size];

    int originalWidth = paddedWidth - (padding*2);
    int originalHeight = paddedHeight - (padding*2);

    // first batch loading preparation
    int dest = threadIdx.y * BLOCK_WIDTH + threadIdx.x;
    int destY = dest / size;
    int destX = dest % size;
    int srcY = blockIdx.y * BLOCK_WIDTH + destY;
    int srcX = blockIdx.x * BLOCK_WIDTH + destX;

    // second batch loading preparation
    int dest_2 = threadIdx.y * BLOCK_WIDTH + threadIdx.x + BLOCK_WIDTH * BLOCK_WIDTH;
    int destY_2 = dest_2 / size;
    int destX_2 = dest_2 % size;
    int srcY_2 = blockIdx.y * BLOCK_WIDTH + destY_2;
    int srcX_2 = blockIdx.x * BLOCK_WIDTH + destX_2;

    // TODO uncomment for BLOCK_WIDTH == 8
    // third batch loading preparation
//    int dest_3 = threadIdx.y * BLOCK_WIDTH + threadIdx.x + (BLOCK_WIDTH * BLOCK_WIDTH * 2);
//    int destY_3 = dest_3 / size;
//    int destX_3 = dest_3 % size;
//    int srcY_3 = blockIdx.y * BLOCK_WIDTH + destY_3 - padding;
//    int srcX_3 = blockIdx.x * BLOCK_WIDTH + destX_3 - padding;

    // first batch loading
    int src = (srcY * paddedWidth + srcX) * numChannels;
    if (srcY >= 0 && srcY < paddedHeight && srcX >= 0 && srcX < paddedWidth) {
        dataR_ds[destY][destX] = flatPaddedImage[src];
        dataG_ds[destY][destX] = flatPaddedImage[src + 1];
        dataB_ds[destY][destX] = flatPaddedImage[src + 2];
    }

    // second batch loading
    src = (srcY_2 * paddedWidth + srcX_2) * numChannels;
    if (destY_2 < size) {
        if (srcY_2 >= 0 && srcY_2 < paddedHeight && srcX_2 >= 0 && srcX_2 < paddedWidth) {
            dataR_ds[destY_2][destX_2] = flatPaddedImage[src];
            dataG_ds[destY_2][destX_2] = flatPaddedImage[src + 1];
            dataB_ds[destY_2][destX_2] = flatPaddedImage[src + 2];
        }
    }

    // TODO uncomment for BLOCK_WIDTH == 8
    // third batch loading
//    src = (srcY_3 * paddedWidth + srcX_3) * numChannels;
//    if (destY_3 < size) {
//        if (srcY_3 >= 0 && srcY_3 < paddedHeight && srcX_3 >= 0 && srcX_3 < paddedWidth) {
//            dataR_ds[destY_3][destX_3] = flatPaddedImage[src];
//            dataG_ds[destY_3][destX_3] = flatPaddedImage[src + 1];
//            dataB_ds[destY_3][destX_3] = flatPaddedImage[src + 2];
//        }
//    }
    __syncthreads();


    int y = blockIdx.y * BLOCK_WIDTH + threadIdx.y;
    int x = blockIdx.x * BLOCK_WIDTH + threadIdx.x;

    if (y >= padding && y < (originalHeight + padding) && x >= padding && x < (originalWidth + padding)){
        float accumR = 0;
        float accumG = 0;
        float accumB = 0;
        float maskValue;
        for (int i = 0; i < kernelDim; i++) {
            for (int j = 0; j < kernelDim; j++) {
                maskValue = gaussianKernelDevice[i * kernelDim + j];
                accumR += dataR_ds[threadIdx.y + i][threadIdx.x + j] * maskValue;
                accumG += dataG_ds[threadIdx.y + i][threadIdx.x + j] * maskValue;
                accumB += dataB_ds[threadIdx.y + i][threadIdx.x + j] * maskValue;
            }
        }
        flatBlurredImage[(y * paddedWidth + x) * numChannels] = accumR/scalarValue;
        flatBlurredImage[(y * paddedWidth + x) * numChannels + 1] = accumG/scalarValue;
        flatBlurredImage[(y * paddedWidth + x) * numChannels + 2] = accumB/scalarValue;
    }
    __syncthreads();

}

vector<double> CUDASharedKernelTest(int numExecutions, int numBlocks, const FlatPaddedImage& paddedImage, AbstractKernel& kernel) {
    vector<double> meanExecutionsTimeVec;
    double meanExecutionsTime = 0;
    for (int execution = 0; execution < numExecutions; execution++) {
        float *flatPaddedImage;
        float *flatPaddedImageDevice;
        int padding = paddedImage.getPadding();
        int originalWidth = paddedImage.getOriginalWidth();
        int originalHeight = paddedImage.getOriginalHeight();

        int paddedWidth = paddedImage.getOriginalWidth() + (padding * 2);
        int paddedHeight = paddedImage.getOriginalHeight() + (padding * 2);

        int numChannels = paddedImage.getNumChannels();
        int paddedSize = paddedWidth * paddedHeight * numChannels;
        float *flatBlurredImage;
        float *flatBlurredImageDevice;

        float gaussianKernel[max_kernel_size];
        int kernelDim = kernel.getKernelDimension();
        int kernelSize = kernel.getKernelSize();
        float scalarValue = kernel.getScalarValue();

        const int blockWidth = BLOCK_WIDTH;
        const int blockHeight = BLOCK_WIDTH;

        // allocate host memory
        checkCudaShared(cudaMallocHost((void **) &flatPaddedImage, sizeof(float) * paddedSize));
        checkCudaShared(cudaMallocHost((void **) &flatBlurredImage, sizeof(float) * paddedSize));
        checkCudaShared(cudaMallocHost((void **) &gaussianKernel, sizeof(float) * max_kernel_size));

        flatPaddedImage = paddedImage.getFlatPaddedImage();
        float *gaussianKernel_temp = kernel.getFlatKernel();

        // initialize gaussianKernel
        for (int i = 0; i < max_kernel_size; i++) {
            if (i < kernelSize)
                gaussianKernel[i] = gaussianKernel_temp[i];
            else
                gaussianKernel[i] = 0;
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
                cudaMemcpyToSymbol(gaussianKernelDevice, gaussianKernel, sizeof(float) * max_kernel_size, 0,
                                   cudaMemcpyHostToDevice));

        // define DimGrid and DimBlock
        dim3 DimGrid((int) ceil((float) (paddedWidth) / (float) blockWidth),
                     (int) ceil((float) (paddedHeight) / (float) blockHeight));
        dim3 DimBlock(blockWidth, blockHeight);

        auto start = std::chrono::system_clock::now();
        // start shared convolution
        shared_kernel_convolution_3D<<<DimGrid, DimBlock>>>(flatPaddedImageDevice, paddedWidth, paddedHeight,
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