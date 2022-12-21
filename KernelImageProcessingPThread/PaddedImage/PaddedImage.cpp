#include "PaddedImage.h"

PaddedImage::PaddedImage(const String& path, int _padding) {
    originalImage = imread(path, IMREAD_UNCHANGED);
    padding = _padding;
    width = originalImage.cols + padding * 2;
    height = originalImage.rows + padding * 2;
    numChannels = originalImage.channels();
    paddedImage = createPaddedImage();
}

vector<vector<vector<float>>> PaddedImage::createPaddedImage() {
    vector<vector<vector<float>>> imageWithPadding;
    imageWithPadding.resize(height);
    for (int i = 0; i < height; i++) {
        imageWithPadding[i].resize(width);
        for (int j = 0; j < width; j++) {
            imageWithPadding[i][j].resize(numChannels);
        }
    }

    // scroll by rows
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
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
    return imageWithPadding;
}

int PaddedImage::getWidth() const {
    return width;
}

int PaddedImage::getHeight() const {
    return height;
}

int PaddedImage::getNumChannels() const {
    return numChannels;
}

int PaddedImage::getPadding() const {
    return padding;
}

vector<vector<vector<float>>> PaddedImage::getPaddedImage() const {
    return paddedImage;
}
