#ifndef KERNELIMAGEPROCESSING_PARALLELKERNELTEST_H
#define KERNELIMAGEPROCESSING_PARALLELKERNELTEST_H
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

struct kernelProcessingRows_args{
    vector<vector<vector<float>>>* paddedImage;
    vector<vector<vector<float>>>* blurredImage;
    float** kernelMatrix;
    int width;
    int padding;
    int kernelDimension;
    float scalarValue;
    int startIndex_i;
    int endIndex_i;
};

void* applyKernelRows(void *args) {
    auto *arguments = (kernelProcessingRows_args*) args;
    for (int i = arguments->startIndex_i; i <= arguments->endIndex_i; i++) {
        for (int j = arguments->padding; j < arguments->width - arguments->padding; j++) {
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
vector<double> parallelPThreadTestRowsDivision(int numExecutions, int numThreads, const PaddedImage& image, AbstractKernel& kernel){
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
        cout << "Thread number: " << nThread << endl;
        for (int execution = 0; execution < numExecutions; execution++) {
            vector<vector<vector<float>>> blurredImage = image.getPaddedImage();
            vector<pthread_t> threads(nThread);
            vector<kernelProcessingRows_args> arguments(nThread);
            int chuckSizeHeight = floor((height - (padding * 2)) / nThread);
            auto start = chrono::system_clock::now();
            for (int t = 0; t < nThread - 1; t++) {
                arguments[t].paddedImage = &paddedImage;
                arguments[t].blurredImage = &blurredImage;
                arguments[t].kernelMatrix = kernelMatrix;
                arguments[t].width = width;
                arguments[t].padding = padding;
                arguments[t].kernelDimension = kernelDimension;
                arguments[t].scalarValue = scalarValue;
                arguments[t].startIndex_i = chuckSizeHeight * t + padding;
                arguments[t].endIndex_i = chuckSizeHeight * (t + 1) - 1 + padding;
                if (pthread_create(&threads[t], NULL, applyKernelRows, (void *) &arguments[t]) != 0)
                    cout << "Error" << endl;
            }
            arguments[nThread - 1].paddedImage = &paddedImage;
            arguments[nThread - 1].blurredImage = &blurredImage;
            arguments[nThread - 1].kernelMatrix = kernelMatrix;
            arguments[nThread - 1].width = width;
            arguments[nThread - 1].padding = padding;
            arguments[nThread - 1].kernelDimension = kernelDimension;
            arguments[nThread - 1].scalarValue = scalarValue;
            arguments[nThread - 1].startIndex_i = chuckSizeHeight * (nThread - 1) + padding;
            arguments[nThread - 1].endIndex_i = height - padding - 1;
            if (pthread_create(&threads[nThread - 1], NULL, applyKernelRows, (void *) &arguments[nThread - 1]) != 0)
                cout << "Error" << endl;

            for (auto thread: threads) {
                pthread_join(thread, NULL);
            }
            chrono::duration<double> executionTime{};
            executionTime = chrono::system_clock::now() - start;
            auto executionTimeMilliseconds = chrono::duration_cast<chrono::milliseconds>(executionTime);
            meanExecutionsTime += (double) executionTimeMilliseconds.count();
            Mat reconstructed_image1 = imageReconstruction(blurredImage, width, height, padding);
            imwrite("../results/blurredRows_" + to_string(nThread) + ".jpeg", reconstructed_image1);
            blurredImage.clear();
            threads.clear();
            arguments.clear();
        }
        meanExecutionsTimeVec.push_back(meanExecutionsTime / numExecutions);
    }
    return meanExecutionsTimeVec;
}
#endif //KERNELIMAGEPROCESSING_PARALLELKERNELTEST_H
