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
        cout << "-------------------------------------------------" << endl;
        cout << endl;
        cout << "Sequential Test with " << name << "p" << endl;
        double meanExecTimeSequentialTest = sequentialTest(numExecutions, flatPaddedImage, gaussianKernel);
        cout << "Mean Sequential execution time: " << floor(meanExecTimeSequentialTest * 100.) / 100. << " microseconds\n" << endl;

        index = 0;
        vector<vector<double>> timeCudaGlobalTest = CUDAGlobalKernelTest(numExecutions, numBlocks, flatPaddedImage, gaussianKernel);
        vector<double> meanExecTimeCudaGlobalTest = timeCudaGlobalTest[0];
        vector<double> meanCopyTimeCudaGlobalTest = timeCudaGlobalTest[1];
        cout << "-------------------------------------------------" << endl;
        cout << endl;
        cout << "\nCUDA Test Global Kernel with " << name << "p" << endl;
        for (int blockDimension = 2; blockDimension <= numBlocks; blockDimension *= 2) {
            cout << "Mean CUDA execution time with " << blockDimension << " as blockDimension: " << floor(meanExecTimeCudaGlobalTest[index] * 100.) / 100. << " microseconds" << endl;
            cout << "Mean CUDA copy time with " << blockDimension << " as blockDimension: " << floor(meanCopyTimeCudaGlobalTest[index] * 100.) / 100. << " microseconds" << endl;
            cout << endl;
            index++;
        }

        index = 0;
        vector<vector<double>> timeCudaConstantTest = CUDAConstantKernelTest(numExecutions, numBlocks, flatPaddedImage, gaussianKernel);
        vector<double> meanExecTimeCudaConstantTest = timeCudaConstantTest[0];
        vector<double> meanCopyTimeCudaConstantTest = timeCudaConstantTest[1];
        cout << "-------------------------------------------------" << endl;
        cout << endl;
        cout << "\nCUDA Test Constant Kernel with " << name << "p" << endl;
        for (int blockDimension = 2; blockDimension <= numBlocks; blockDimension *= 2) {
            cout << "Mean CUDA execution time with " << blockDimension << " as blockDimension: " << floor(meanExecTimeCudaConstantTest[index] * 100.) / 100. << " microseconds" << endl;
            cout << "Mean CUDA copy time with " << blockDimension << " as blockDimension: " << floor(meanCopyTimeCudaConstantTest[index] * 100.) / 100. << " microseconds" << endl;
            cout << endl;
            index++;
        }

        index = 0;
        vector<vector<double>> timeCudaSharedTest = CUDASharedKernelTest(numExecutions, numBlocks, flatPaddedImage, gaussianKernel);
        vector<double> meanExecTimeCudaSharedTest = timeCudaSharedTest[0];
        vector<double> meanCopyTimeCudaSharedTest = timeCudaSharedTest[1];
        cout << "-------------------------------------------------" << endl;
        cout << endl;
        cout << "\nCUDA Test Shared Kernel with " << name << "p" << endl;
        for (int blockDimension = 2; blockDimension <= numBlocks; blockDimension *= 2) {
            if (blockDimension > 2 * padding){
                cout << "Mean CUDA execution time with " << blockDimension << " as blockDimension: " << floor(meanExecTimeCudaSharedTest[index] * 100.) / 100. << " microseconds" << endl;
                cout << "Mean CUDA copy time with " << blockDimension << " as blockDimension: " << floor(meanCopyTimeCudaSharedTest[index] * 100.) / 100. << " microseconds" << endl;
                cout << endl;
                index++;
            }
        }
    }
    return 0;
}
