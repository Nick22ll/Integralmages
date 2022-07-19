
#ifndef INTEGRALIMAGESCUDA_CUDAINTEGRALIMAGE_CU
#define INTEGRALIMAGESCUDA_CUDAINTEGRALIMAGE_CU

#include "GreyImage.h"
#define BLOCK_DIM 16

// Macro to aligned up to the memory size in question
#define MEMORY_ALIGNMENT  4096
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )


using namespace std;

//Check CUDA Errors

cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}


//Kernels
__global__ void CUDASumRowIntegralImage(int width, int height, int* integralImage){

    //each thread is responsible for a specific row
    int thread_ID = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_ID < height){
        //Fill the row properly (each cell is a sum with the previous one)
        for(int col=1; col < width; col++)
            integralImage[thread_ID * width + col] += integralImage[thread_ID * width + (col-1)];
    }
}

__global__ void transpose(int *odata, int *idata, int width, int height)
{
    //the shared memory array is sized to (BLOCK_DIM+1)*BLOCK_DIM --> this pads each row of the 2D block in shared memory
    // so that bank conflicts do not occur when threads address the array column-wise.
    __shared__ float block[BLOCK_DIM][BLOCK_DIM+1];

    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    // synchronise to ensure all writes to block[][] have completed
    __syncthreads();

    // write the transposed matrix tile to global memory (odata) in linear order
    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = block[threadIdx.x][threadIdx.y];
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
    int* d_transpose;

    int block_num_for_columns = (width + thread_per_block)/thread_per_block;
    int block_num_for_rows = (height+thread_per_block)/thread_per_block;

    //checkCuda(cudaHostRegister(&original_image, sizeof(int) * size, cudaHostRegisterDefault));
    //checkCuda(cudaHostRegister(&parallel_integral_image, sizeof(int) * size, cudaHostRegisterDefault));

    checkCuda(cudaMallocHost((void**)&original_image, sizeof(int) * size));
    checkCuda(cudaMallocHost((void**)&parallel_integral_image, sizeof(int) * size));

    int* tmp = image.getImage();
    for(int i=0;i<height;i++)
        for(int j=0; j<width; j++)
            original_image[i*width+j] = tmp[i*width+j];
    delete[] tmp;

    //Allocate device memory
    checkCuda(cudaMalloc((void**)&d_original_image, sizeof(int) * size));
    checkCuda(cudaMalloc((void**)&d_integral_image, sizeof(int) * size));
    checkCuda(cudaMalloc((void**)&d_transpose, sizeof(int) * size));

    // Transfer data from host to device memory
    checkCuda(cudaMemcpy(d_original_image, original_image, sizeof(int) * size, cudaMemcpyHostToDevice));

    // Executing kernel

    dim3 grid((width + BLOCK_DIM)/ BLOCK_DIM, (height+BLOCK_DIM)/ BLOCK_DIM, 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 grid_after_trasp((height+BLOCK_DIM)/ BLOCK_DIM, (width + BLOCK_DIM)/BLOCK_DIM, 1);

    transpose<<<grid,threads>>>(d_transpose, d_original_image, width, height);
    CUDASumRowIntegralImage<<<block_num_for_columns, thread_per_block>>>(height, width, d_transpose);
    transpose<<<grid_after_trasp,threads>>>(d_integral_image,d_transpose, height, width);
    CUDASumRowIntegralImage<<<block_num_for_rows, thread_per_block>>>(width, height, d_integral_image);

    // Transfer data back to host memory
    checkCuda(cudaMemcpy(parallel_integral_image, d_integral_image, sizeof(int) * size, cudaMemcpyDeviceToHost));

    // Deallocate device memory
    checkCuda(cudaFree(d_original_image));
    checkCuda(cudaFree(d_integral_image));
    checkCuda(cudaFree(d_transpose));

    checkCuda(cudaFreeHost(original_image));
    //delete[] original_image;

    return parallel_integral_image;
}

double timeParallelIntegralImage(GreyImage image, const int thread_per_block){
    int height = image.getHeight();
    int width = image.getWidth();

    //Allocate host memory
    //int* original_image = image.getImage();
    int* original_image;
    int size = height*width;
    //int* parallel_integral_image = new int[size];
    int* parallel_integral_image;

    int* d_original_image;
    int* d_integral_image;
    int* d_transpose;

    int block_num_for_columns = (width + thread_per_block)/thread_per_block;
    int block_num_for_rows = (height+thread_per_block)/thread_per_block;

    //checkCuda(cudaHostRegister(&original_image, sizeof(int) * size, cudaHostRegisterDefault));
    //checkCuda(cudaHostRegister(&parallel_integral_image, sizeof(int) * size, cudaHostRegisterDefault));

    checkCuda(cudaMallocHost((void**)&original_image, sizeof(int) * size));
    checkCuda(cudaMallocHost((void**)&parallel_integral_image, sizeof(int) * size));

    int* tmp = image.getImage();
    for(int i=0;i<height;i++)
        for(int j=0; j<width; j++)
            original_image[i*width+j] = tmp[i*width+j];
    delete[] tmp;

    dim3 grid((width + BLOCK_DIM)/ BLOCK_DIM, (height+BLOCK_DIM)/ BLOCK_DIM, 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 grid_after_trasp((height+BLOCK_DIM)/ BLOCK_DIM, (width + BLOCK_DIM)/BLOCK_DIM, 1);

    //Allocate device memory
    checkCuda(cudaMalloc((void**)&d_original_image, sizeof(int) * size));
    checkCuda(cudaMalloc((void**)&d_integral_image, sizeof(int) * size));
    checkCuda(cudaMalloc((void**)&d_transpose, sizeof(int) * size));

    // Transfer data from host to device memory
    checkCuda(cudaMemcpy(d_original_image, original_image, sizeof(int) * size, cudaMemcpyHostToDevice));

    //warm-up
    transpose<<<grid,threads>>>(d_transpose, d_original_image, width, height);
    CUDASumRowIntegralImage<<<block_num_for_columns, thread_per_block>>>(height, width, d_transpose);
    transpose<<<grid_after_trasp,threads>>>(d_integral_image,d_transpose, height, width);
    CUDASumRowIntegralImage<<<block_num_for_rows, thread_per_block>>>(width, height, d_integral_image);
    cudaDeviceSynchronize();

    // Executing kernels
    auto start = std::chrono::system_clock::now();

    transpose<<<grid,threads>>>(d_transpose, d_original_image, width, height);
    CUDASumRowIntegralImage<<<block_num_for_columns, thread_per_block>>>(height, width, d_transpose);
    transpose<<<grid_after_trasp,threads>>>(d_integral_image,d_transpose, height, width);
    CUDASumRowIntegralImage<<<block_num_for_rows, thread_per_block>>>(width, height, d_integral_image);
    cudaDeviceSynchronize();

    std::chrono::duration<double> diff{};
    diff = std::chrono::system_clock::now() - start;

    // Transfer data back to host memory
    checkCuda(cudaMemcpy(parallel_integral_image, d_integral_image, sizeof(int) * size, cudaMemcpyDeviceToHost));

    // Deallocate device memory
    checkCuda(cudaFree(d_original_image));
    checkCuda(cudaFree(d_integral_image));
    checkCuda(cudaFree(d_transpose));

    cudaFreeHost(original_image);
    cudaFreeHost(parallel_integral_image);
    //delete[] parallel_integral_image;
    //delete[] original_image;

    return diff.count();
}

#endif //INTEGRALIMAGESCUDA_CUDAINTEGRALIMAGE_CU
