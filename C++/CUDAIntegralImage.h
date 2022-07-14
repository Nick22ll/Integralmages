//
// Created by Nicco on 14/07/2022.
//

#ifndef INTEGRALIMAGESCUDA_CUDAINTEGRALIMAGE_CU
#define INTEGRALIMAGESCUDA_CUDAINTEGRALIMAGE_CU

#include "GreyImage.h"

using namespace std;

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

    cudaHostRegister(&original_image, size, cudaHostRegisterReadOnly);
    cudaHostRegister(&parallel_integral_image, size, cudaHostRegisterDefault);

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


    // Deallocate device memory
    cudaFree(d_original_image);
    cudaFree(d_integral_image);
    cudaFree(d_height);
    cudaFree(d_width);

    return parallel_integral_image;
}



double timeParallelIntegralImage(GreyImage image, const int thread_per_block){
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

    cudaHostRegister(&original_image, size, cudaHostRegisterReadOnly);
    cudaHostRegister(&parallel_integral_image, size, cudaHostRegisterDefault);

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
    auto start = std::chrono::system_clock::now();

    CUDAfillColumnIntegralImage<<<block_num_for_columns, thread_per_block>>>(d_original_image, d_height, d_width, d_integral_image);
    CUDAfillRowIntegralImage<<<block_num_for_rows, thread_per_block>>>(d_height, d_width, d_integral_image);

    cudaDeviceSynchronize();
    std::chrono::duration<double> diff{};
    diff = std::chrono::system_clock::now() - start;

    // Transfer data back to host memory
    cudaMemcpy(parallel_integral_image, d_integral_image, sizeof(int) * width * height, cudaMemcpyDeviceToHost);


    // Deallocate device memory
    cudaFree(d_original_image);
    cudaFree(d_integral_image);
    cudaFree(d_height);
    cudaFree(d_width);

    delete[] parallel_integral_image;
    return diff.count();
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

    //Register host memory for improved memcopy speed
    //cudaHostRegister(&original_image, size, cudaHostRegisterReadOnly);
    //cudaHostRegister(&parallel_integral_image, size, cudaHostRegisterDefault);

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

#endif //INTEGRALIMAGESCUDA_CUDAINTEGRALIMAGE_CU
