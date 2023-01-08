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
    int numExecutions = 100;
    int numThreads = 16;
    int index;
//    GaussianKernel3x3 gaussianKernel = GaussianKernel3x3();
    GaussianKernel5x5 gaussianKernel = GaussianKernel5x5();
    int padding = floor(gaussianKernel.getKernelDimension()/2);
    vector<string> folder_names = {"480", "720", "1080", "2K", "4K"};
    for(const auto& name: folder_names){
        string path = "../images/" + name + "/image.jpg";
        PaddedImage paddedImage = PaddedImage(path, padding);
        cout << "Sequential Test with " << name << "p" << endl;
        double meanExecTimeSequentialTest = sequentialTest(numExecutions, paddedImage, gaussianKernel);
        cout << "Mean Sequential execution time: " << floor(meanExecTimeSequentialTest * 100.) / 100. << " microseconds\n" << endl;

        index = 0;
        cout << "PThread Test Rows Division with " << name << "p" << endl;
        vector<double> meanExecTimePThreadTestRows = parallelPThreadTestRowsDivision(numExecutions, numThreads, paddedImage, gaussianKernel);
        for(int nThread = 2; nThread <= numThreads; nThread+=2) {
            cout << "Mean PThread Rows Division execution time with " << nThread << " thread: " << floor(meanExecTimePThreadTestRows[index] * 100.) / 100. << " microseconds" << endl;
            index++;
        }

        index = 0;
        cout << "\nPThread Test Rows and Columns Division with " << name << "p" << endl;
        vector<double> meanExecTimePThreadTestRowsColumns = parallelPThreadTestRowsColumnsDivision(numExecutions, numThreads, paddedImage, gaussianKernel);
        for(int nThread = 2; nThread <= numThreads; nThread+=2) {
            cout << "Mean PThread Rows and Columns Division execution time with " << nThread << " thread: " << floor(meanExecTimePThreadTestRowsColumns[index] * 100.) / 100. << " microseconds" << endl;
            index++;
        }

        cout << "\nPThread Test Blocks Division with " << name << "p" << endl;
        int block_dims[] = {4, 8, 16, 32, 64, 128, 256};
        for(int block_dim : block_dims){
            index = 0;
            cout << "Block size: " << block_dim << endl;
            vector<double> meanExecTimePThreadTestBlocks = parallelPThreadTestBlocks(numExecutions, numThreads, paddedImage, gaussianKernel, block_dim);
            for(int nThread = 2; nThread <= numThreads; nThread+=2) {
                int width = paddedImage.getWidth();
                int height = paddedImage.getHeight();
                int blockRows;
                int blockColumns;
                if(height % block_dim == 0)
                    blockRows = height / block_dim;
                else
                    blockRows = (int)(trunc(height / block_dim)) + 1;
                if(width % block_dim == 0)
                    blockColumns = width / block_dim;
                else
                    blockColumns = (int)(trunc(width / block_dim)) + 1;
                int numBlocks = blockRows * blockColumns;
                if (numBlocks >= nThread) {
                    cout << "Mean PThread Blocks execution time with " << nThread << " thread: " << floor(meanExecTimePThreadTestBlocks[index] * 100.) / 100. << " microseconds" << endl;
                    index++;
                }
            }
            cout << "\n";
        }
    }
    return 0;
}