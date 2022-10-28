#ifndef KERNELIMAGEPROCESSING_PARALLELKERNELTEST_H
#define KERNELIMAGEPROCESSING_PARALLELKERNELTEST_H
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "../PaddedImage/PaddedImage.h"
#include "../Kernel/Kernel.h"
#include <pthread.h>
#include <vector>
#include <chrono>
#include "../PaddedImage/imageReconstruction.h"
#include "cmath"

using namespace cv;
using namespace std;

struct kernelProcessing_args{
    vector<vector<vector<float>>>* paddedImage;
    vector<vector<vector<float>>>* blurredImage;
    int** kernelMatrix;
    int width;
    int height;
    int padding;
    int kernelSize;
    int scalarValue;
    int startIndex_i;
    int endIndex_i;
    int startIndex_j;
    int endIndex_j;
};

void* applyKernel(void *args) {
    auto *arguments = (kernelProcessing_args*) args;
    for (int i = arguments->startIndex_i; i <= arguments->endIndex_i; i++) {
        for (int j = arguments->padding; j < arguments->width - arguments->padding; j++) {
            float newValueR = 0;
            float newValueG = 0;
            float newValueB = 0;
            for (int k = -arguments->padding; k < arguments->kernelSize - arguments->padding; k++) {
                for (int l = -arguments->padding; l < arguments->kernelSize - arguments->padding; l++) {
                    newValueR += arguments->paddedImage->at(i + k).at(j + l).at(0) * arguments->kernelMatrix[k + arguments->padding][l + arguments->padding];
                    newValueG += arguments->paddedImage->at(i + k).at(j + l).at(1) * arguments->kernelMatrix[k + arguments->padding][l + arguments->padding];
                    newValueB += arguments->paddedImage->at(i + k).at(j + l).at(2) * arguments->kernelMatrix[k + arguments->padding][l + arguments->padding];
                }
            }
            arguments->blurredImage->at(i).at(j).at(0) = newValueR / arguments->scalarValue;
            arguments->blurredImage->at(i).at(j).at(1) = newValueG / arguments->scalarValue;
            arguments->blurredImage->at(i).at(j).at(2) = newValueB / arguments->scalarValue;
        }
    }
}
vector<double> parallelPThreadTest(int numExecutions, int numThreads, const PaddedImage& image, Kernel& kernel){
    int** kernelMatrix = kernel.getKernel();
    int scalarValue = kernel.getScalarValue();
    int kernelSize = kernel.getKernelSize();
    int padding = image.getPadding();
    int width = image.getWidth();
    int height = image.getHeight();
    vector<vector<vector<float>>> paddedImage = image.getPaddedImage();
    vector<double> meanExecutionsTimeVec;
    for(int nThread = 2; nThread <= numThreads; nThread++) {
        double meanExecutionsTime = 0;
        for (int execution = 0; execution < numExecutions; execution++) {
            vector<vector<vector<float>>> blurredImage = paddedImage;
            vector<pthread_t> threads(nThread);
            vector<kernelProcessing_args> arguments(nThread);
            auto start = chrono::system_clock::now();
            int chuckSizeHeight = floor((height - (padding * 2)) / nThread);
            for (int t = 0; t < nThread - 1; t++) {
                arguments[t].paddedImage = &paddedImage;
                arguments[t].blurredImage = &blurredImage;
                arguments[t].kernelMatrix = kernelMatrix;
                arguments[t].width = width;
                arguments[t].height = height;
                arguments[t].padding = padding;
                arguments[t].kernelSize = kernelSize;
                arguments[t].scalarValue = scalarValue;
                arguments[t].startIndex_i = chuckSizeHeight * t + padding;
                arguments[t].endIndex_i = chuckSizeHeight * (t + 1) - 1 + padding;
                if (pthread_create(&threads[t], NULL, applyKernel, (void*)&arguments[t]) != 0)
                    cout << "Error" << endl;
            }
            arguments[nThread - 1].paddedImage = &paddedImage;
            arguments[nThread - 1].blurredImage = &blurredImage;
            arguments[nThread - 1].kernelMatrix = kernelMatrix;
            arguments[nThread - 1].width = width;
            arguments[nThread - 1].height = height;
            arguments[nThread - 1].padding = padding;
            arguments[nThread - 1].kernelSize = kernelSize;
            arguments[nThread - 1].scalarValue = scalarValue;
            arguments[nThread - 1].startIndex_i = chuckSizeHeight * (nThread - 1) + padding;
            arguments[nThread - 1].endIndex_i = height - padding - 1;
            if (pthread_create(&threads[nThread - 1], NULL, applyKernel, (void*)&arguments[nThread - 1]) != 0)
                cout << "Error" << endl;

            for (auto thread:threads) {
                pthread_join(thread, NULL);
            }

            chrono::duration<double> executionTime{};
            executionTime = chrono::system_clock::now() - start;
            auto executionTimeMilliseconds = chrono::duration_cast<chrono::milliseconds>(executionTime);
            meanExecutionsTime += (double) executionTimeMilliseconds.count();
//            Mat reconstructed_image1 = imageReconstruction(blurredImage, width, height, padding);
//            imwrite("../blurred.jpeg", reconstructed_image1);
//            Mat reconstructed_image2 = imageReconstruction(paddedImage, width, height, padding);
//            imwrite("../original.jpeg", reconstructed_image2);
            blurredImage.clear();
            threads.clear();
            arguments.clear();
        }
        cout << "Iteration: " << nThread << endl;
        meanExecutionsTimeVec.push_back(meanExecutionsTime / numExecutions);
    }
    return meanExecutionsTimeVec;
}
#endif //KERNELIMAGEPROCESSING_PARALLELKERNELTEST_H
