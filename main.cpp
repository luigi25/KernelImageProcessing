#include <iostream>
#include "PaddedImage/PaddedImage.h"
#include "PaddedImage/imageReconstruction.h"
#include "Kernel/prova.h"
#include "KernelProcessingTests/sequentialKernelTest.h"
#include "KernelProcessingTests/parallelKernelTest.h"
#include "Kernel/Kernel.h"
using namespace cv;
using namespace std;


int main(){
    // Read the images
//    PaddedImage paddedImage = PaddedImage("../images/image.jpg", 9);
//    int kernelDimension = 3;
//    int kernelScalarValue = 16;
//    int **kernel = new int*[kernelDimension];
//    kernel[0] = new int[kernelDimension] {1, 2, 1};
//    kernel[1] = new int[kernelDimension] {2, 4, 2};
//    kernel[2] = new int[kernelDimension] {1, 2, 1};
//    Kernel GaussianKernel3x3 = Kernel(kernel, kernelDimension, kernelScalarValue);

    PaddedImage paddedImage = PaddedImage("../images/image.jpg", 25);
    int kernelDimension = 5;
    int kernelScalarValue = 256;
    int **kernel = new int*[kernelDimension];
    kernel[0] = new int[kernelDimension] {1, 4, 6, 4, 1};
    kernel[1] = new int[kernelDimension] {4, 16, 24, 16, 4};
    kernel[2] = new int[kernelDimension] {6, 24, 36, 24, 6};
    kernel[3] = new int[kernelDimension] {4, 16, 24, 16, 4};
    kernel[4] = new int[kernelDimension] {1, 4, 6, 4, 1};
    Kernel GaussianKernel5x5 = Kernel(kernel, kernelDimension, kernelScalarValue);
    int numExecutions = 10;

//    cout << "Sequential Test" << endl;
//    double meanExecTimeSequentialTest = sequentialTest(numExecutions, paddedImage, GaussianKernel5x5);
//    cout << "Mean Sequential execution time: " << floor(meanExecTimeSequentialTest * 100.) / 100. << " milliseconds\n" << endl;

    cout << "PThread Test" << endl;
    int numThreads = 10;
    vector<double> meanExecTimePThreadTest = parallelPThreadTest(numExecutions, numThreads, paddedImage, GaussianKernel5x5);
    for(int nThread = 1; nThread <= numThreads; nThread++) {
        cout << "Mean PThread execution time with " << nThread << " thread: " << floor(meanExecTimePThreadTest[nThread - 1] * 100.) / 100. << " milliseconds" << endl;
    }

//    Mat reconstructed_image1 = imageReconstruction(prova, paddedImage.getWidth(), paddedImage.getHeight(), paddedImage.getPadding());
//    imwrite("blur.bmp", reconstructed_image1);

//    Mat reconstructed_image1 = imageReconstruction(imgA1.getPaddedImage(), imgA1.getWidth(), imgA1.getHeight(), imgA1.getPadding());
//    Mat reconstructed_image2 = imageReconstruction(imgA2.getPaddedImage(), imgA2.getWidth(), imgA2.getHeight(), imgA2.getPadding());
//    namedWindow("Padded Image - padding 1", WINDOW_NORMAL);
//    resizeWindow("Padded Image - padding 1", (int)(imgA1.getWidth()*0.35), (int)(imgA1.getHeight()*0.35));
//    imshow("Padded Image - padding 1", reconstructed_image1);
//    waitKey(0);
    return 0;
}