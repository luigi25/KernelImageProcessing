#ifndef KERNELIMAGEPROCESSING_PADDEDIMAGE_H
#define KERNELIMAGEPROCESSING_PADDEDIMAGE_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

class PaddedImage {
private:
    int width, height, numChannels, padding;
    vector<vector<vector<float>>> paddedImage;
    Mat originalImage;

public:
    explicit PaddedImage(const String& path, int _padding);
    vector<vector<vector<float>>> createPaddedImage();
    int getWidth() const;
    int getHeight() const;
    int getNumChannels() const;
    int getPadding() const;
    vector<vector<vector<float>>> getPaddedImage() const;

};


#endif //KERNELIMAGEPROCESSING_PADDEDIMAGE_H
