#include <iostream>
#include "PaddedImage/PaddedImage.h"
#include "PaddedImage/imageReconstruction.h"
#include "KernelProcessingTests/sequentialKernelTest.h"
#include "KernelProcessingTests/parallelKernelTest.h"
#include "Kernel/GaussianKernel5x5.h"
#include "Kernel/GaussianKernel3x3.h"
using namespace cv;
using namespace std;


int main(){
    // Read the images
//    GaussianKernel3x3 gaussianKernel3x3 = GaussianKernel3x3();
    GaussianKernel5x5 gaussianKernel5x5 = GaussianKernel5x5();
    PaddedImage paddedImage = PaddedImage("../images/image.jpg", gaussianKernel5x5.getPadding());
    int numExecutions = 100;
    int numThreads = 10;

    cout << "Sequential Test" << endl;
    double meanExecTimeSequentialTest = sequentialTest(numExecutions, paddedImage, gaussianKernel5x5);
    cout << "Mean Sequential execution time: " << floor(meanExecTimeSequentialTest * 100.) / 100. << " milliseconds\n" << endl;

    cout << "PThread Test" << endl;
    vector<double> meanExecTimePThreadTest = parallelPThreadTest(numExecutions, numThreads, paddedImage, gaussianKernel5x5);
    for(int nThread = 2; nThread <= numThreads; nThread++) {
        cout << "Mean PThread execution time with " << nThread << " thread: " << floor(meanExecTimePThreadTest[nThread - 2] * 100.) / 100. << " milliseconds" << endl;
    }


//    Mat reconstructed_image1 = imageReconstruction(imgA1.getPaddedImage(), imgA1.getWidth(), imgA1.getHeight(), imgA1.getPadding());
//    Mat reconstructed_image2 = imageReconstruction(imgA2.getPaddedImage(), imgA2.getWidth(), imgA2.getHeight(), imgA2.getPadding());
//    namedWindow("Padded Image - padding 1", WINDOW_NORMAL);
//    resizeWindow("Padded Image - padding 1", (int)(imgA1.getWidth()*0.35), (int)(imgA1.getHeight()*0.35));
//    imshow("Padded Image - padding 1", reconstructed_image1);
//    waitKey(0);
    return 0;
}