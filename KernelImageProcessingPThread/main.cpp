#include <iostream>
#include "PaddedImage/PaddedImage.h"
#include "PaddedImage/imageReconstruction.h"
#include "KernelProcessingTests/sequentialKernelTest.h"
#include "KernelProcessingTests/parallelKernelTestRowsDivision.h"
#include "KernelProcessingTests/parallelKernelTestRowsColumnsDivision.h"
#include "KernelProcessingTests/parallelKernelTestBlocks.h"
#include "Kernel/GaussianKernel5x5.h"
#include "Kernel/GaussianKernel3x3.h"
using namespace cv;
using namespace std;


int main(){
    // Read the images
//    GaussianKernel3x3 gaussianKernel = GaussianKernel3x3();
    GaussianKernel5x5 gaussianKernel = GaussianKernel5x5();
    int numExecutions = 10;
    int numThreads = 16;
    vector<string> folder_names = {"480", "720", "1080", "2K", "4K"};
    for(const auto& name: folder_names){
        PaddedImage paddedImage = PaddedImage("../images/" + name + "/image.jpg", floor(gaussianKernel.getKernelDimension()/2));
        cout << "Sequential Test with " << name << "p" << endl;
        double meanExecTimeSequentialTest = sequentialTest(numExecutions, paddedImage, gaussianKernel);
        cout << "Mean Sequential execution time: " << floor(meanExecTimeSequentialTest * 100.) / 100. << " milliseconds\n" << endl;

        cout << "PThread Test Rows Division with " << name << "p" << endl;
        vector<double> meanExecTimePThreadTestRows = parallelPThreadTestRowsDivision(numExecutions, numThreads, paddedImage, gaussianKernel);
        for(int nThread = 2; nThread <= numThreads; nThread+=2) {
            cout << "Mean PThread Rows Division execution time with " << nThread << " thread: " << floor(meanExecTimePThreadTestRows[nThread/2 - 1] * 100.) / 100. << " milliseconds" << endl;
        }

        cout << "\nPThread Test Rows and Columns Division with " << name << "p" << endl;
        vector<double> meanExecTimePThreadTestRowsColumns = parallelPThreadTestRowsColumnsDivision(numExecutions, numThreads, paddedImage, gaussianKernel);
        for(int nThread = 4; nThread <= numThreads; nThread+=2) {
            cout << "Mean PThread Rows and Columns Division execution time with " << nThread << " thread: " << floor(meanExecTimePThreadTestRowsColumns[nThread/2 - 2] * 100.) / 100. << " milliseconds" << endl;
        }

        cout << "\nPThread Test Blocks Division with " << name << "p" << endl;
        int block_dims[] = {8, 16, 32, 64, 128, 256, 512};
        for(int block_dim : block_dims){
            cout << "Block size: " << block_dim << endl;
            vector<double> meanExecTimePThreadTestBlocks = parallelPThreadTestBlocks(numExecutions, numThreads, paddedImage, gaussianKernel, block_dim);
            for(int nThread = 2; nThread <= numThreads; nThread+=2) {
                cout << "Mean PThread Blocks execution time with " << nThread << " thread: " << floor(meanExecTimePThreadTestBlocks[nThread/2 - 1] * 100.) / 100. << " milliseconds" << endl;
            }
            cout << "\n";
        }
    }

//    Mat reconstructed_image1 = imageReconstruction(imgA1.getPaddedImage(), imgA1.getWidth(), imgA1.getHeight(), imgA1.getPadding());
//    Mat reconstructed_image2 = imageReconstruction(imgA2.getPaddedImage(), imgA2.getWidth(), imgA2.getHeight(), imgA2.getPadding());
//    namedWindow("Padded Image - padding 1", WINDOW_NORMAL);
//    resizeWindow("Padded Image - padding 1", (int)(imgA1.getWidth()*0.35), (int)(imgA1.getHeight()*0.35));
//    imshow("Padded Image - padding 1", reconstructed_image1);
//    waitKey(0);
    return 0;
}