//
// Created by Nicco on 14/07/2022.
//

#ifndef INTEGRALIMAGESCUDA_TESTS_H
#define INTEGRALIMAGESCUDA_TESTS_H

#include "SequentialIntegralImage.h"
#include "CUDAIntegralImage.h"

using namespace std;



vector<double> testSequential(GreyImage& image, int executions = 100){
    vector<double> mean_values;;
    double min_exec_time= 10000 , max_exec_time = 0, mean_exec_time = 0;

    for(int execution=0; execution<executions; execution++){
        double exec_time = timeSequentialIntegralImage(image);
        mean_exec_time += exec_time;
        //Check min
        if(exec_time < min_exec_time)
            min_exec_time = exec_time;

        //Check max
        if(exec_time > max_exec_time)
            max_exec_time = exec_time;
    }

    mean_exec_time /= executions;
    mean_values.push_back(mean_exec_time);

    cout << "Sequential Minimum Execution Time: " << min_exec_time << " seconds!"<<endl;
    cout << "Sequential Maximum Execution Time: " << max_exec_time << " seconds!"<<endl;
    cout << "Sequential Mean Execution Time: " << mean_exec_time<< " seconds!"<<endl;


    return mean_values;
}

vector<double> testSequentialwithMemory(GreyImage& image, int executions = 100){
    vector<double> mean_values;
    std::chrono::duration<double> diff{};
    double min_exec_time= 10000 , max_exec_time = 0, mean_exec_time = 0;

    for(int execution=0; execution<executions; execution++){
        auto start = std::chrono::system_clock::now();
        sequentialIntegralImage(image);
        diff = std::chrono::system_clock::now() - start;
        double exec_time = diff.count();
        mean_exec_time += exec_time;
        //Check min
        if(exec_time < min_exec_time)
            min_exec_time = exec_time;

        //Check max
        if(exec_time > max_exec_time)
            max_exec_time = exec_time;
    }

    mean_exec_time /= executions;
    mean_values.push_back(mean_exec_time);

    cout << "Sequential Minimum Execution Time: " << min_exec_time << " seconds!"<<endl;
    cout << "Sequential Maximum Execution Time: " << max_exec_time << " seconds!"<<endl;
    cout << "Sequential Mean Execution Time: " << mean_exec_time<< " seconds!"<<endl;


    return mean_values;
}



vector<double> testCUDAwithMemory(const GreyImage& image, const vector<int>& num_threads, int executions = 100){
    vector<double> mean_values;
    std::chrono::duration<double> diff{};

    for(auto threads : num_threads){

        double min_exec_time= 10000 , max_exec_time = 0, mean_exec_time = 0;

        for(int execution=0; execution<executions; execution++){
            auto start = std::chrono::system_clock::now();
            int* a = parallelIntegralImage(image, threads);
            diff = std::chrono::system_clock::now() - start;
            double exec_time = diff.count();
            mean_exec_time += exec_time;

            delete[] a;

            //Check min
            if(exec_time < min_exec_time)
                min_exec_time = exec_time;

            //Check max
            if(exec_time > max_exec_time)
                max_exec_time = exec_time;
        }

        mean_exec_time /= executions;
        mean_values.push_back(mean_exec_time);

        cout << "\nTEST with " << threads << " threads:" <<endl;
        cout << "Parallel Minimum Execution Time: " << min_exec_time << " seconds!"<<endl;
        cout << "Parallel Maximum Execution Time: " << max_exec_time << " seconds!"<<endl;
        cout << "Parallel Mean Execution Time: " << mean_exec_time<< " seconds!"<<endl;
    }

    return mean_values;
}

vector<double> testCUDA(const GreyImage& image, const vector<int>& num_threads, int executions = 100){
    vector<double> mean_values;

    for(auto threads : num_threads){

        double min_exec_time= 10000 , max_exec_time = 0, mean_exec_time = 0;

        for(int execution=0; execution<executions; execution++){
            double exec_time = timeParallelIntegralImage(image, threads);
            mean_exec_time += exec_time;

            //Check min
            if(exec_time < min_exec_time)
                min_exec_time = exec_time;

            //Check max
            if(exec_time > max_exec_time)
                max_exec_time = exec_time;
        }

        mean_exec_time /= executions;
        mean_values.push_back(mean_exec_time);

        cout << "\nTEST with " << threads << " threads:" <<endl;
        cout << "Parallel Minimum Execution Time: " << min_exec_time << " seconds!"<<endl;
        cout << "Parallel Maximum Execution Time: " << max_exec_time << " seconds!"<<endl;
        cout << "Parallel Mean Execution Time: " << mean_exec_time<< " seconds!"<<endl;
    }

    return mean_values;
}





#endif //INTEGRALIMAGESCUDA_TESTS_H
