#include <iostream>
#include "GreyImage.h"
#include "Tests.h"

using namespace std;


int main() {

    GreyImage image = GreyImage(R"(U:\Magistrale\Parallel Computing\IntegralImagesCUDA\images\big.jpg)");

    //cout << "STARTING SEQUENTIAL TEST! \n" << endl;
    //testSequential(image);

    vector<int> num_threads = {256};
    cout << "STARTING PARALLEL TEST! \n"<< endl;
    testCUDA(image, num_threads);
    testCUDAwithMemory(image, num_threads);


    return 0;
}


