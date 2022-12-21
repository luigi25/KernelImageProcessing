#include <iostream>
#include "PaddedImage/FlatPaddedImage.h"
#include "Kernel/GaussianKernel5x5.h"
#include "Kernel/GaussianKernel3x3.h"
#include "KernelProcessingTests/CUDAKernelTestGlobal.h"
#include "KernelProcessingTests/SequentialKernelTest.h"
using namespace cv;
using namespace std;

int main() {
    //    GaussianKernel3x3 gaussianKernel = GaussianKernel3x3();
    GaussianKernel5x5 gaussianKernel = GaussianKernel5x5();
    vector<string> folder_names = { "720", "1080", "2K", "4K"};
    for(const auto& name: folder_names){
        FlatPaddedImage flatPaddedImage = FlatPaddedImage("../images/" + name + "/image.jpg", floor(gaussianKernel.getKernelDimension() / 2));
        int numExecutions = 100;

        cout << "Sequential Test with " << name << "p" << endl;
        double meanExecTimeSequentialTest = sequentialTest(numExecutions, flatPaddedImage, gaussianKernel);
        cout << "Mean Sequential execution time: " << floor(meanExecTimeSequentialTest * 100.) / 100. << " milliseconds\n" << endl;

        int index = 0;
        vector<double> meanExecTimeCudaTest = parallelCudaTest(numExecutions, flatPaddedImage, gaussianKernel);
        cout << "\nCUDA Test with " << name << "p" << endl;
        for (int blockDimension = 2; blockDimension <= 32; blockDimension *= 2) {
            cout << "Mean CUDA execution time with " << blockDimension << " as blockDimension: " << floor(meanExecTimeCudaTest[index] * 100.) / 100. << " microseconds" << endl;
            index++;
        }
    }
    return 0;
}
