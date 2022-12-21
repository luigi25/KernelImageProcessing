#ifndef KERNELIMAGEPROCESSING_PADDEDIMAGE_H
#define KERNELIMAGEPROCESSING_PADDEDIMAGE_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

class FlatPaddedImage {
private:
    int originalWidth, originalHeight, numChannels, padding;
    float* createFlatPaddedImage();
    float* flatPaddedImage;
    Mat originalImage;

public:
    explicit FlatPaddedImage(const String& path, int _padding);
    int getOriginalWidth() const;
    int getOriginalHeight() const;
    int getNumChannels() const;
    int getPadding() const;
    float* getFlatPaddedImage() const;

};


#endif //KERNELIMAGEPROCESSING_PADDEDIMAGE_H
