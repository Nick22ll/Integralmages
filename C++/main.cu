#include <iostream>
#include "GreyImage.h"
#include "Tests.h"

using namespace std;


int main() {

    vector<string> image_names = { "480", "720", "1080", "1440", "2160"};

    for(const auto& name: image_names){
        GreyImage image = GreyImage("../images/" + name + ".jpg");

        cout << "STARTING SEQUENTIAL TEST WITH IMAGE: " <<name<< ".jpg! \n" << endl;
        double mean_sequential = testSequential(image);

        cout << "STARTING MEMORY SEQUENTIAL TEST WITH IMAGE: " <<name<< ".jpg! \n" << endl;
        double mean_sequential_memory = testSequentialwithMemory(image);

        vector<int> num_threads = {2, 4, 8, 16, 32, 64, 128};

        cout << "STARTING PARALLEL TEST WITH IMAGE: " <<name<< ".jpg! \n"<< endl;
        vector<double> mean_values = testCUDA(image, num_threads);
        for(int i=0;i<num_threads.size();i++)
            cout<< "Speedup with " << num_threads[i] << " threads is: " << mean_sequential/mean_values[i] << endl;

        cout << "STARTING MEMORY PARALLEL TEST WITH IMAGE: " <<name<< ".jpg! \n"<< endl;
        vector<double> memory_mean_values = testCUDAwithMemory(image, num_threads);
        for(int i=0;i<num_threads.size();i++)
            cout<< "Speedup with " << num_threads[i] << " threads is: " << mean_sequential_memory/memory_mean_values[i] << endl;
    }
    return 0;
}


