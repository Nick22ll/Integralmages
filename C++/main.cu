#include <iostream>
#include "GreyImage.h"

using namespace std;

int** sequentialIntegralImage(GreyImage originalImage){
        int width = originalImage.getWidth();
        int height = originalImage.getHeight();
        int **integralImage = new int*[height];
        for(int i=0; i<height;i++)
            integralImage[i] = new int[width];

        //Fill the first Cell
        integralImage[0][0] = originalImage.getPixel(0,0);

        //Fill the first column
        for(int i = 1; i < height ; i++)
            integralImage[i][0] = integralImage[i - 1][0] + originalImage.getPixel(0,i);

        //Fill the first row
        for(int j = 1; j < width ; j++)
            integralImage[0][j] = integralImage[0][j-1] + originalImage.getPixel(j, 0);

        for(int i=1; i < height; i++){
            for(int j=1; j < width ; j++){
                integralImage[i][j] = originalImage.getPixel(j,i) + integralImage[i][j-1] + integralImage[i-1][j] - integralImage[i-1][j-1];
            }
        }

        return integralImage;
}


__global__ void CUDAfillColumnIntegralImage(int* originalImage, int *d_height, int *d_width, int* integralImage){

    //each thread is responsible for a specific column
    int thread_ID = blockIdx.x * blockDim.x + threadIdx.x;
    int original_height = (*d_height);
    int original_width = (*d_width);

    if(thread_ID < original_width){
        //Fill the first cell of a column (which is simple copy of the original cell)
        integralImage[0 * original_width + thread_ID] = originalImage[0 * original_width + thread_ID];
        //Fill the column properly
        for(int row=1; row<original_height; row++)
            integralImage[row * original_width + thread_ID] = integralImage[(row - 1)* original_width + thread_ID] + originalImage[row * original_width + thread_ID];
    }

}

__global__ void CUDAfillRowIntegralImage(int *d_height, int *d_width, int* integralImage){

    //each thread is responsible for a specific row
    int thread_ID = blockIdx.x * blockDim.x + threadIdx.x;
    int original_height = (*d_height);
    int original_width = (*d_width);

    if(thread_ID < original_height){
        //Fill the row properly (each cell is a sum with the previous one)
        for(int col=1; col < original_width; col++)
            integralImage[thread_ID * original_width + col] += integralImage[thread_ID * original_width + (col-1)];
    }
}


int* parallelIntegralImage(GreyImage image, const int thread_per_block){
    int height = image.getHeight();
    int width = image.getWidth();

    //Allocate host memory
    int* original_image = image.getImage();
    int size = height*width;
    int* parallel_integral_image = new int[size];

    int* d_original_image;
    int* d_integral_image;
    int* d_height;
    int* d_width;


    int block_num_for_columns = (width + thread_per_block)/thread_per_block;
    int block_num_for_rows = (height+thread_per_block)/thread_per_block;

    //Allocate device memory
    cudaMalloc((void**)&d_original_image, sizeof(int) * width * height);
    cudaMalloc((void**)&d_integral_image, sizeof(int) * width * height);
    cudaMalloc((void**)&d_height, sizeof(int));
    cudaMalloc((void**)&d_width, sizeof(int));


    // Transfer data from host to device memory
    cudaMemcpy(d_original_image, original_image, sizeof(int) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_integral_image, parallel_integral_image, sizeof(int) * width * height , cudaMemcpyHostToDevice);
    cudaMemcpy(d_height, &height, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_width, &width, sizeof(int), cudaMemcpyHostToDevice);

    // Executing kernel
    CUDAfillColumnIntegralImage<<<block_num_for_columns, thread_per_block>>>(d_original_image, d_height, d_width, d_integral_image);
    CUDAfillRowIntegralImage<<<block_num_for_rows, thread_per_block>>>(d_height, d_width, d_integral_image);

    // Transfer data back to host memory
    cudaMemcpy(parallel_integral_image, d_integral_image, sizeof(int) * width * height, cudaMemcpyDeviceToHost);

    cudaFree(d_original_image);
    cudaFree(d_integral_image);
    cudaFree(d_height);
    cudaFree(d_width);

    return parallel_integral_image;
}


int* parallelStreamIntegralImage(GreyImage image, const int thread_per_block){
    int height = image.getHeight();
    int width = image.getWidth();

    //Allocate host memory
    int* original_image = image.getImage();
    int size = height*width;
    int* parallel_integral_image = new int[size];

    int* d_original_image;
    int* d_integral_image;
    int* d_height;
    int* d_width;


    int block_num_for_columns = (width + thread_per_block)/thread_per_block;
    int block_num_for_rows = (height+thread_per_block)/thread_per_block;


    cudaStream_t stream1, stream2, stream3, stream4;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    //Allocate device memory
    cudaMalloc((void**)&d_original_image, sizeof(int) * width * height);
    cudaMalloc((void**)&d_integral_image, sizeof(int) * width * height);
    cudaMalloc((void**)&d_height, sizeof(int));
    cudaMalloc((void**)&d_width, sizeof(int));


    // Transfer data from host to device memory
    cudaMemcpyAsync(d_original_image, original_image, sizeof(int) * width * height, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_integral_image, parallel_integral_image, sizeof(int) * width * height , cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_height, &height, sizeof(int), cudaMemcpyHostToDevice, stream3);
    cudaMemcpyAsync(d_width, &width, sizeof(int), cudaMemcpyHostToDevice, stream4);

    // Executing kernel
    CUDAfillColumnIntegralImage<<<block_num_for_columns, thread_per_block,0, stream3>>>(d_original_image, d_height, d_width, d_integral_image);
    CUDAfillRowIntegralImage<<<block_num_for_rows, thread_per_block, 0, stream3>>>(d_height, d_width, d_integral_image);

    // Transfer data back to host memory
    cudaMemcpyAsync(parallel_integral_image, d_integral_image, sizeof(int) * width * height, cudaMemcpyDeviceToHost, stream1);

    cudaFree(d_original_image);
    cudaFree(d_integral_image);
    cudaFree(d_height);
    cudaFree(d_width);

    return parallel_integral_image;
}




vector<double> testSequential(const GreyImage& image, int executions = 100){
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



vector<double> testCUDA(const GreyImage& image, const vector<int>& num_threads, int executions = 100){
    vector<double> mean_values;
    std::chrono::duration<double> diff{};

    for(auto threads : num_threads){

        double min_exec_time= 10000 , max_exec_time = 0, mean_exec_time = 0;

        for(int execution=0; execution<executions; execution++){
            auto start = std::chrono::system_clock::now();
            parallelIntegralImage(image, threads);
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

        cout << "\nTEST with " << threads << " threads:" <<endl;
        cout << "Parallel Minimum Execution Time: " << min_exec_time << " seconds!"<<endl;
        cout << "Parallel Maximum Execution Time: " << max_exec_time << " seconds!"<<endl;
        cout << "Parallel Mean Execution Time: " << mean_exec_time<< " seconds!"<<endl;
    }

    return mean_values;
}



int main() {

    GreyImage image = GreyImage(R"(U:\Magistrale\Parallel Computing\IntegralImagesCUDA\images\big.jpg)");

    cout << "STARTING SEQUENTIAL TEST! \n" << endl;
    testSequential(image);

    vector<int> num_threads = {256};
    cout << "STARTING PARALLEL TEST! \n"<< endl;
    testCUDA(image, num_threads);


    return 0;
}


