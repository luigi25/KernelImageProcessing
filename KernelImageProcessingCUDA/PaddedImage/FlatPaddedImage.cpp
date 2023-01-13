#include "FlatPaddedImage.h"

FlatPaddedImage::FlatPaddedImage(const String& path, int _padding) {
    originalImage = imread(path, IMREAD_UNCHANGED);
    padding = _padding;
    originalWidth = originalImage.cols;
    originalHeight = originalImage.rows;
    numChannels = originalImage.channels();
    flatPaddedImage = createFlatPaddedImage();
}

float* FlatPaddedImage::createFlatPaddedImage() {
    int paddedWidth = originalWidth + (padding * 2);
    int paddedHeight = originalHeight + (padding * 2);
    vector<vector<vector<float>>> imageWithPadding;
    imageWithPadding.resize(paddedHeight);
    for (int i = 0; i < paddedHeight; i++) {
        imageWithPadding[i].resize(paddedWidth);
        for (int j = 0; j < paddedWidth; j++) {
            imageWithPadding[i][j].resize(numChannels);
        }
    }

    // scroll by rows
    for (int i = 0; i < paddedHeight; i++) {
        for (int j = 0; j < paddedWidth; j++) {
            int index_i = i - padding;
            if (index_i < 0) {
                index_i = 0;
            } else if(index_i >= originalImage.rows) {
                index_i = originalImage.rows - 1;
            }
            int index_j = j - padding;
            if (index_j < 0){
                index_j = 0;
            } else if (index_j >= originalImage.cols) {
                index_j = originalImage.cols - 1;
            }

            Vec3b pixel = originalImage.at<Vec3b>(index_i, index_j);
            imageWithPadding[i][j][0] = (float)pixel[2];
            imageWithPadding[i][j][1] = (float)pixel[1];
            imageWithPadding[i][j][2] = (float)pixel[0];
        }
    }
    int size = paddedWidth * paddedHeight * numChannels;
    float* flatImage = new float[size];

    // scroll by rows
    for(int i = 0; i < paddedHeight; i++) {
        for (int j = 0; j < paddedWidth; j++) {
            flatImage[(i * paddedWidth * numChannels) + (j * numChannels)] = imageWithPadding[i][j][0];
            flatImage[(i * paddedWidth * numChannels) + (j * numChannels) + 1] = imageWithPadding[i][j][1];
            flatImage[(i * paddedWidth * numChannels) + (j * numChannels) + 2] = imageWithPadding[i][j][2];
        }
    }
    return flatImage;
}

int FlatPaddedImage::getOriginalWidth() const {
    return originalWidth;
}

int FlatPaddedImage::getOriginalHeight() const {
    return originalHeight;
}

int FlatPaddedImage::getNumChannels() const {
    return numChannels;
}

int FlatPaddedImage::getPadding() const {
    return padding;
}

float* FlatPaddedImage::getFlatPaddedImage() const {
    return flatPaddedImage;
}
