#ifndef KERNELIMAGEPROCESSING_PROVA_H
#define KERNELIMAGEPROCESSING_PROVA_H

#include <vector>

using namespace std;

const int** GaussianKernel3x3() {
    const int dimension = 3;
    const int **GaussianKernel = new const int*[dimension];

    GaussianKernel[0] = new const int[dimension] {1, 2, 1};
    GaussianKernel[1] = new const int[dimension] {2, 4, 2};
    GaussianKernel[2] = new const int[dimension] {1, 2, 1};

    return GaussianKernel;
}

const int** GaussianKernel5x5() {
    const int dimension = 5;
    const int **GaussianKernel = new const int*[dimension];

    GaussianKernel[0] = new const int[dimension] {1, 4, 6, 4, 1};
    GaussianKernel[1] = new const int[dimension] {4, 16, 24, 16, 4};
    GaussianKernel[2] = new const int[dimension] {6, 24, 36, 24, 6};
    GaussianKernel[3] = new const int[dimension] {4, 16, 24, 16, 4};
    GaussianKernel[4] = new const int[dimension] {1, 4, 6, 4, 1};

    return GaussianKernel;
}
#endif //KERNELIMAGEPROCESSING_PROVA_H
