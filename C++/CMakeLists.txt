cmake_minimum_required(VERSION 3.22)
project(IntegralImagesCUDA CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)


find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(IntegralImagesCUDA main.cu GreyImage.cpp GreyImage.h SequentialIntegralImage.h CUDAIntegralImage.h Tests.h)
target_link_libraries( IntegralImagesCUDA ${OpenCV_LIBS})
set_target_properties(IntegralImagesCUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
