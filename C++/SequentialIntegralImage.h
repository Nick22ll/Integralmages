//
// Created by Nicco on 14/07/2022.
//

#ifndef INTEGRALIMAGESCUDA_SEQUENTIALINTEGRALIMAGE_H
#define INTEGRALIMAGESCUDA_SEQUENTIALINTEGRALIMAGE_H

#include "GreyImage.h"

using namespace std;

int** sequentialIntegralImage(GreyImage& originalImage){

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

    for(int i=1; i < height; i++)
        for(int j=1; j < width ; j++)
            integralImage[i][j] = originalImage.getPixel(j,i) + integralImage[i][j-1] + integralImage[i-1][j] - integralImage[i-1][j-1];

    return integralImage;
}

double timeSequentialIntegralImage(GreyImage& originalImage){
    int width = originalImage.getWidth();
    int height = originalImage.getHeight();
    int **integralImage = new int*[height];
    for(int i=0; i<height;i++)
        integralImage[i] = new int[width];

    auto start = std::chrono::system_clock::now();

    //Fill the first Cell
    integralImage[0][0] = originalImage.getPixel(0,0);

    //Fill the first column
    for(int i = 1; i < height ; i++)
        integralImage[i][0] = integralImage[i - 1][0] + originalImage.getPixel(0,i);

    //Fill the first row
    for(int j = 1; j < width ; j++)
        integralImage[0][j] = integralImage[0][j-1] + originalImage.getPixel(j, 0);

    for(int i=1; i < height; i++)
        for(int j=1; j < width ; j++)
            integralImage[i][j] = originalImage.getPixel(j,i) + integralImage[i][j-1] + integralImage[i-1][j] - integralImage[i-1][j-1];

    std::chrono::duration<double> diff{};
    diff = std::chrono::system_clock::now() - start;

    for(int i=0; i<height;i++)
        delete[] integralImage[i];
    delete[] integralImage;

    return diff.count();
}

#endif //INTEGRALIMAGESCUDA_SEQUENTIALINTEGRALIMAGE_H
