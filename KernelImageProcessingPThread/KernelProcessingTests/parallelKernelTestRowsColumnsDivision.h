#ifndef KERNELIMAGEPROCESSING_PARALLELKERNELTESTROWSCOLUMNSDIVISION_H
#define KERNELIMAGEPROCESSING_PARALLELKERNELTESTROWSCOLUMNSDIVISION_H
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


struct StartEndRowColIndices{
    vector<int> startIndex_i;
    vector<int> endIndex_i;
    vector<int> startIndex_j;
    vector<int> endIndex_j;
};

StartEndRowColIndices createStartEndIndicesRowColForChunk(int width, int height, int padding, int chuckSizeWidth, int chuckSizeHeight, int threadRows, int threadColumns){
    // set start/end indices for image pixel and each thread
    StartEndRowColIndices rowsColsIndices;
    for (int r = 0; r < threadRows; r++) {
        for (int c = 0; c < threadColumns; c++) {
            if (r != threadRows - 1){
                rowsColsIndices.startIndex_i.push_back(chuckSizeHeight * r + padding);
                rowsColsIndices.endIndex_i.push_back(chuckSizeHeight * (r + 1) - 1 + padding);
            } else {
                rowsColsIndices.startIndex_i.push_back(chuckSizeHeight * (threadRows - 1) + padding);
                rowsColsIndices.endIndex_i.push_back(height - padding - 1);
            }
        }
    }

    for (int r = 0; r < threadRows; r++) {
        for (int c = 0; c < threadColumns - 1; c++) {
            rowsColsIndices.startIndex_j.push_back(chuckSizeWidth * c + padding);
            rowsColsIndices.endIndex_j.push_back(chuckSizeWidth * (c + 1) - 1 + padding);
        }
        rowsColsIndices.startIndex_j.push_back(chuckSizeWidth * (threadColumns - 1) + padding);
        rowsColsIndices.endIndex_j.push_back(width - padding - 1);
    }
    return rowsColsIndices;
}

struct kernelProcessing_args{
    vector<vector<vector<float>>>* paddedImage;
    vector<vector<vector<float>>>* blurredImage;
    float** kernelMatrix;
    int padding;
    int kernelDimension;
    float scalarValue;
    int startIndex_i;
    int endIndex_i;
    int startIndex_j;
    int endIndex_j;
};

void* applyKernelRowsColumns(void *args) {
    // apply filtering
    auto *arguments = (kernelProcessing_args*) args;
    for (int i = arguments->startIndex_i; i <= arguments->endIndex_i; i++) {
        for (int j = arguments->startIndex_j; j <= arguments->endIndex_j; j++) {
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
    return (void*)("Done!");
}
vector<double> parallelPThreadTestRowsColumnsDivision(int numExecutions, int numThreads, const PaddedImage& image, AbstractKernel& kernel){
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
        // define threads per rows and columns
        int threadRows;
        int threadColumns;
        if (nThread == 2 || nThread == 4 || nThread % 4 != 0) {
            threadRows = nThread / 2;
            threadColumns = nThread / threadRows;
        }
        else {
            threadColumns = nThread / 4;
            threadRows = nThread / threadColumns;
        }
        // define chunk size
        int chuckSizeHeight = floor((height - (padding * 2)) / threadRows);
        int chuckSizeWidth = floor((width - (padding * 2)) / threadColumns);
        StartEndRowColIndices rowsColsIndices = createStartEndIndicesRowColForChunk(width, height, padding, chuckSizeWidth, chuckSizeHeight, threadRows, threadColumns);
        for (int execution = 0; execution < numExecutions; execution++) {
            // create the output image
            vector<vector<vector<float>>> blurredImage = image.getPaddedImage();
            vector<pthread_t> threads(nThread);
            vector<kernelProcessing_args> arguments(nThread);
            auto start = chrono::system_clock::now();
            // pass arguments for each thread
            for (int t = 0; t < nThread; t++) {
                arguments[t].paddedImage = &paddedImage;
                arguments[t].blurredImage = &blurredImage;
                arguments[t].kernelMatrix = kernelMatrix;
                arguments[t].padding = padding;
                arguments[t].kernelDimension = kernelDimension;
                arguments[t].scalarValue = scalarValue;
                arguments[t].startIndex_i = rowsColsIndices.startIndex_i[t];
                arguments[t].endIndex_i = rowsColsIndices.endIndex_i[t];
                arguments[t].startIndex_j = rowsColsIndices.startIndex_j[t];
                arguments[t].endIndex_j = rowsColsIndices.endIndex_j[t];
                if (pthread_create(&threads[t], NULL, applyKernelRowsColumns, (void *) &arguments[t]) != 0)
                    cout << "Error" << endl;
            }

            for (auto thread: threads) {
                pthread_join(thread, NULL);
            }
            chrono::duration<double> executionTime{};
            executionTime = chrono::system_clock::now() - start;
            auto executionTimeMicroseconds = chrono::duration_cast<chrono::microseconds>(executionTime);
            meanExecutionsTime += (double) executionTimeMicroseconds.count();
//            Mat reconstructed_image1 = imageReconstruction(blurredImage, width, height, padding);
//            imwrite("../results/blurredRowsColumns_" + to_string(nThread) + ".jpeg", reconstructed_image1);
            blurredImage.clear();
            threads.clear();
            arguments.clear();
        }
        meanExecutionsTimeVec.push_back(meanExecutionsTime / numExecutions);
    }
    return meanExecutionsTimeVec;
}
#endif //KERNELIMAGEPROCESSING_PARALLELKERNELTESTROWSCOLUMNSDIVISION_H
