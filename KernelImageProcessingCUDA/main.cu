#include <iostream>
#include "PaddedImage/FlatPaddedImage.h"
#include "Kernel/GaussianKernel5x5.h"
#include "Kernel/GaussianKernel3x3.h"
#include "KernelProcessingTests/CUDAKernelTestGlobal.h"
#include "KernelProcessingTests/CUDAKernelTestConstant.h"
#include "KernelProcessingTests/CUDAKernelTestShared.h"
#include "KernelProcessingTests/SequentialKernelTest.h"
using namespace cv;
using namespace std;

int main() {
    int numExecutions = 100;
    int numBlocks = 32;
    int index;
//    GaussianKernel3x3 gaussianKernel = GaussianKernel3x3();
    GaussianKernel5x5 gaussianKernel = GaussianKernel5x5();
    int padding = floor(gaussianKernel.getKernelDimension() / 2);
    vector<string> folder_names = {"480", "720", "1080", "2K", "4K"};
    for(const auto& name: folder_names){
        string path = "../images/" + name + "/image.jpg";
        FlatPaddedImage flatPaddedImage = FlatPaddedImage(path, padding);

        // Sequential test
        cout << "-------------------------------------------------" << endl;
        cout << endl;
        cout << "Sequential Test with " << name << "p" << endl;
        double meanExecTimeSequentialTest = sequentialTest(numExecutions, flatPaddedImage, gaussianKernel);
        cout << "Mean Sequential execution time: " << floor(meanExecTimeSequentialTest * 100.) / 100. << " microseconds\n" << endl;

        // CUDA Global Memory test
        index = 0;
        vector<double> meanExecTimeCudaGlobalTest = CUDAGlobalKernelTest(numExecutions, numBlocks, flatPaddedImage, gaussianKernel);
        cout << "-------------------------------------------------" << endl;
        cout << endl;
        cout << "\nCUDA Test Global Memory with " << name << "p" << endl;
        for (int blockDimension = 2; blockDimension <= numBlocks; blockDimension *= 2) {
            cout << "Mean CUDA execution time with " << blockDimension << " as blockDimension: " << floor(meanExecTimeCudaGlobalTest[index] * 100.) / 100. << " microseconds" << endl;
            cout << endl;
            index++;
        }

        // CUDA Constant Memory test
        index = 0;
        vector<double> meanExecTimeCudaConstantTest = CUDAConstantKernelTest(numExecutions, numBlocks, flatPaddedImage, gaussianKernel);
        cout << "-------------------------------------------------" << endl;
        cout << endl;
        cout << "\nCUDA Test Constant Memory with " << name << "p" << endl;
        for (int blockDimension = 2; blockDimension <= numBlocks; blockDimension *= 2) {
            cout << "Mean CUDA execution time with " << blockDimension << " as blockDimension: " << floor(meanExecTimeCudaConstantTest[index] * 100.) / 100. << " microseconds" << endl;
            cout << endl;
            index++;
        }

        // CUDA Shared Memory test
        index = 0;
        vector<double> meanExecTimeCudaSharedTest = CUDASharedKernelTest(numExecutions, numBlocks, flatPaddedImage, gaussianKernel);
        cout << "-------------------------------------------------" << endl;
        cout << endl;
        cout << "\nCUDA Test Shared Memory with " << name << "p" << endl;
        cout << "Mean CUDA execution time with " << numBlocks << " as blockDimension: " << floor(meanExecTimeCudaSharedTest[index] * 100.) / 100. << " microseconds" << endl;
        cout << endl;
    }
    return 0;
}
