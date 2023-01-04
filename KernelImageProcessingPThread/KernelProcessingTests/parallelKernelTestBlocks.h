#ifndef KERNELIMAGEPROCESSINGPTHREAD_PARALLELKERNELTESTBLOCKS_H
#define KERNELIMAGEPROCESSINGPTHREAD_PARALLELKERNELTESTBLOCKS_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "../PaddedImage/PaddedImage.h"
#include "../Kernel/AbstractKernel.h"
#include <pthread.h>
#include <vector>
#include <chrono>
#include "../PaddedImage/imageReconstruction.h"
#include "cmath"

using namespace cv;
using namespace std;


struct StartEndBlocksIndices{
    vector<int> startIndex_i;
    vector<int> endIndex_i;
    vector<int> startIndex_j;
    vector<int> endIndex_j;
};

vector<StartEndBlocksIndices> createStartEndBlocksIndices(int nThread, int block_dim, int width, int height, int padding){
    StartEndBlocksIndices tempIndices;
    int rows;
    int columns;
    if(height % block_dim == 0)
        rows = height / block_dim;
    else
        rows = (int)(trunc(height / block_dim)) + 1;
    if(width % block_dim == 0)
        columns = width / block_dim;
    else
        columns = (int)(trunc(width / block_dim)) + 1;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < columns; c++) {
            if (r != rows - 1){
                tempIndices.startIndex_i.push_back(block_dim * r + padding);
                tempIndices.endIndex_i.push_back(block_dim * (r + 1) - 1 + padding);
            } else {
                tempIndices.startIndex_i.push_back(block_dim * (rows - 1) + padding);
                tempIndices.endIndex_i.push_back(height - padding - 1);
            }
        }
    }

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < columns - 1; c++) {
            tempIndices.startIndex_j.push_back(block_dim * c + padding);
            tempIndices.endIndex_j.push_back(block_dim * (c + 1) - 1 + padding);
        }
        tempIndices.startIndex_j.push_back(block_dim * (columns - 1) + padding);
        tempIndices.endIndex_j.push_back(width - padding - 1);
    }

    int numBlocks = (int)tempIndices.startIndex_i.size();
    int numBlocksPerThread = (int)numBlocks/nThread;
    vector<int> indicesVector;
    for (int t = 0; t < nThread; t++){
        indicesVector.push_back(numBlocksPerThread * t);
    }
    indicesVector.push_back(numBlocks);
    vector<StartEndBlocksIndices> threadBlocksindices;
    for (int t = 0; t < nThread; t++) {
        StartEndBlocksIndices blocksIndices;
        for (int k = indicesVector[t]; k < indicesVector[t+1]; k++){
            blocksIndices.startIndex_i.push_back(tempIndices.startIndex_i[k]);
            blocksIndices.endIndex_i.push_back(tempIndices.endIndex_i[k]);
            blocksIndices.startIndex_j.push_back(tempIndices.startIndex_j[k]);
            blocksIndices.endIndex_j.push_back(tempIndices.endIndex_j[k]);
        }
        threadBlocksindices.push_back(blocksIndices);
    }
    return threadBlocksindices;
}

struct kernelProcessingBlocks_args{
    vector<vector<vector<float>>>* paddedImage;
    vector<vector<vector<float>>>* blurredImage;
    float** kernelMatrix;
    int padding;
    int kernelDimension;
    float scalarValue;
    StartEndBlocksIndices threadBlocksIndices;
};

void* applyKernelBlocks(void *args) {
    auto *arguments = (kernelProcessingBlocks_args*) args;
    int numbBlocks = (int)arguments->threadBlocksIndices.startIndex_i.size();
    StartEndBlocksIndices threadBlocks = arguments->threadBlocksIndices;
    for (int b = 0; b < numbBlocks; b++){
        for (int i = arguments->threadBlocksIndices.startIndex_i[b]; i <= arguments->threadBlocksIndices.endIndex_i[b]; i++) {
            for (int j = arguments->threadBlocksIndices.startIndex_j[b]; j <= arguments->threadBlocksIndices.endIndex_j[b]; j++) {
                float newValueR = 0;
                float newValueG = 0;
                float newValueB = 0;
                for (int k = -arguments->padding; k < arguments->kernelDimension - arguments->padding; k++) {
                    for (int l = -arguments->padding; l < arguments->kernelDimension - arguments->padding; l++) {
                        newValueR += (*arguments->paddedImage)[i + k][j + l][0] * arguments->kernelMatrix[k + arguments->padding][l + arguments->padding];
                        newValueG += (*arguments->paddedImage)[i + k][j + l][1] * arguments->kernelMatrix[k + arguments->padding][l + arguments->padding];
                        newValueB += (*arguments->paddedImage)[i + k][j + l][2] * arguments->kernelMatrix[k + arguments->padding][l + arguments->padding];
                    }
                }
                (*arguments->blurredImage)[i][j][0] = newValueR / arguments->scalarValue;
                (*arguments->blurredImage)[i][j][1] = newValueG / arguments->scalarValue;
                (*arguments->blurredImage)[i][j][2] = newValueB / arguments->scalarValue;
            }
        }
    }
    return (void*)("Done!");
}
vector<double> parallelPThreadTestBlocks(int numExecutions, int numThreads, const PaddedImage& image, AbstractKernel& kernel, int block_dim){
    float** kernelMatrix = kernel.getKernel();
    float scalarValue = kernel.getScalarValue();
    int kernelDimension = kernel.getKernelDimension();
    int padding = image.getPadding();
    int width = image.getWidth();
    int height = image.getHeight();
    vector<vector<vector<float>>> paddedImage = image.getPaddedImage();
    vector<double> meanExecutionsTimeVec;
    for(int nThread = 2; nThread <= numThreads; nThread+=2) {
        double meanExecutionsTime = 0;
//        cout << "Thread number: " << nThread << endl;
        vector<StartEndBlocksIndices> threadBlocksindices = createStartEndBlocksIndices(nThread, block_dim, width, height, padding);
        for (int execution = 0; execution < numExecutions; execution++) {
            vector<vector<vector<float>>> blurredImage = image.getPaddedImage();
            vector<pthread_t> threads(nThread);
            vector<kernelProcessingBlocks_args> arguments(nThread);
            auto start = chrono::system_clock::now();
            for (int t = 0; t < nThread; t++) {
                arguments[t].paddedImage = &paddedImage;
                arguments[t].blurredImage = &blurredImage;
                arguments[t].kernelMatrix = kernelMatrix;
                arguments[t].padding = padding;
                arguments[t].kernelDimension = kernelDimension;
                arguments[t].scalarValue = scalarValue;
                arguments[t].threadBlocksIndices = threadBlocksindices[t];
                if (pthread_create(&threads[t], NULL, applyKernelBlocks, (void *) &arguments[t]) != 0)
                    cout << "Error" << endl;
            }

            for (auto thread: threads) {
                pthread_join(thread, NULL);
            }
            chrono::duration<double> executionTime{};
            executionTime = chrono::system_clock::now() - start;
            auto executionTimeMilliseconds = chrono::duration_cast<chrono::milliseconds>(executionTime);
            meanExecutionsTime += (double) executionTimeMilliseconds.count();
//            Mat reconstructed_image1 = imageReconstruction(blurredImage, width, height, padding);
//            imwrite("../results/blurredBlock_" + to_string(nThread) + ".jpeg", reconstructed_image1);
            blurredImage.clear();
            threads.clear();
            arguments.clear();
        }
        meanExecutionsTimeVec.push_back(meanExecutionsTime / numExecutions);
    }
    return meanExecutionsTimeVec;
}
#endif //KERNELIMAGEPROCESSINGPTHREAD_PARALLELKERNELTESTBLOCKS_H
