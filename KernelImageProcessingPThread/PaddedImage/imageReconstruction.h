#ifndef KERNELIMAGEPROCESSING_IMAGERECONSTRUCTION_H
#define KERNELIMAGEPROCESSING_IMAGERECONSTRUCTION_H
#include "../PaddedImage/PaddedImage.h"

Mat imageReconstruction(vector<vector<vector<float>>> paddedImg, int width, int height, int padding){
    Mat reconstructedImage = Mat(height - padding * 2, width - padding * 2, CV_8UC3);
    for(int i = padding; i < height - padding; i++) {
        for (int j = padding; j < width - padding; j++) {
            Vec3b pixel = Vec3b((unsigned char)paddedImg[i][j][2], (unsigned char)paddedImg[i][j][1], (unsigned char)paddedImg[i][j][0]);
            reconstructedImage.at<Vec3b>(i - padding,j - padding) = pixel;
        }
    }
    return reconstructedImage;
}

Mat MatImage(vector<vector<vector<float>>> paddedImg, int width, int height){
    Mat MatImage = Mat(height, width, CV_8UC3);
    for(int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            Vec3b pixel = Vec3b((unsigned char)paddedImg[i][j][2], (unsigned char)paddedImg[i][j][1], (unsigned char)paddedImg[i][j][0]);
            MatImage.at<Vec3b>(i,j) = pixel;
        }
    }
    return MatImage;
}
#endif //KERNELIMAGEPROCESSING_IMAGERECONSTRUCTION_H
