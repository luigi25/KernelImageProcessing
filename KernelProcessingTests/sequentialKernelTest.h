#ifndef KERNELIMAGEPROCESSING_SEQUENTIALKERNELTEST_H
#define KERNELIMAGEPROCESSING_SEQUENTIALKERNELTEST_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "../PaddedImage/PaddedImage.h"
#include "../Kernel/Kernel.h"

using namespace cv;
using namespace std;

double sequentialTest(int numExecutions, const PaddedImage& image, Kernel& kernel){
    double meanExecutionsTime = 0;
    int** kernelMatrix = kernel.getKernel();
    int scalarValue = kernel.getScalarValue();
    int kernelSize = kernel.getKernelSize();
    int padding = image.getPadding();
    int width = image.getWidth();
    int height = image.getHeight();
    vector<vector<vector<float>>> paddedImage = image.getPaddedImage();
    for (int execution = 0; execution < numExecutions; execution++) {
        vector<vector<vector<float>>> blurredImage = PaddedImage(image).getPaddedImage();
        auto start = chrono::system_clock::now();
        for (int i = padding; i < height - padding; i++) {
            for (int j = padding; j < width - padding; j++) {
                float newValueR = 0;
                float newValueG = 0;
                float newValueB = 0;
                for (int k = -padding; k < kernelSize - padding; k++) {
                    for (int l = -padding; l < kernelSize - padding; l++) {
                        newValueR += paddedImage[i + k][j + l][0] * kernelMatrix[k + padding][l + padding];
                        newValueG += paddedImage[i + k][j + l][1] * kernelMatrix[k + padding][l + padding];
                        newValueB += paddedImage[i + k][j + l][2] * kernelMatrix[k + padding][l + padding];
                    }
                }
                blurredImage[i][j][0] = newValueR / scalarValue;
                blurredImage[i][j][1] = newValueG / scalarValue;
                blurredImage[i][j][2] = newValueB / scalarValue;
            }
        }
        chrono::duration<double> executionTime{};
        executionTime = chrono::system_clock::now() - start;
        auto executionTimeMilliseconds = chrono::duration_cast<chrono::milliseconds>(executionTime);
        meanExecutionsTime += (double) executionTimeMilliseconds.count();
        blurredImage.clear(); //TODO free memory
//        cout << "Iteration: " << execution << endl;

    }
//    return blurredImage;
    return meanExecutionsTime / numExecutions;
}
#endif //KERNELIMAGEPROCESSING_SEQUENTIALKERNELTEST_H
