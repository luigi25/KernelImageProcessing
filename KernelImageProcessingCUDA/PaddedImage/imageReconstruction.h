#ifndef KERNELIMAGEPROCESSING_IMAGERECONSTRUCTION_H
#define KERNELIMAGEPROCESSING_IMAGERECONSTRUCTION_H
#include "FlatPaddedImage.h"

Mat imageReconstruction(float* flatPaddedImg, int width, int height, int channels, int padding){
    Mat reconstructedImage = Mat(height, width, CV_8UC3);
    vector<vector<vector<float>>> paddedImg;
    int paddedWidth = width + (padding * 2);
    int paddedHeight = height + (padding * 2);
    paddedImg.resize(paddedHeight);
    for (int i = 0; i < paddedHeight; i++) {
        paddedImg[i].resize(paddedWidth);
        for (int j = 0; j < paddedWidth; j++) {
            paddedImg[i][j].resize(channels);
        }
    }

    for(int i = 0; i < paddedHeight; i++) {
        for (int j = 0; j < paddedWidth; j++) {
            paddedImg[i][j][0] = flatPaddedImg[(i * paddedWidth * channels) + (j * channels)];
            paddedImg[i][j][1] = flatPaddedImg[(i * paddedWidth * channels) + (j * channels) + 1];
            paddedImg[i][j][2] = flatPaddedImg[(i * paddedWidth * channels) + (j * channels) + 2];
        }
    }

    for(int i = padding; i < paddedHeight - padding; i++) {
        for (int j = padding; j < paddedWidth - padding; j++) {
            Vec3b pixel = Vec3b((unsigned char)paddedImg[i][j][2], (unsigned char)paddedImg[i][j][1], (unsigned char)paddedImg[i][j][0]);
            reconstructedImage.at<Vec3b>(i - padding, j - padding) = pixel;
        }
    }
    return reconstructedImage;
}
#endif //KERNELIMAGEPROCESSING_IMAGERECONSTRUCTION_H
