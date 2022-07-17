//
// Created by Nicco on 11/07/2022.
//

#include "GreyImage.h"
#include <iostream>

using namespace cv;

GreyImage::GreyImage(String path) {
    //convert RGB image in BW
    image = imread(path, IMREAD_GRAYSCALE);

    width = image.cols;
    height= image.rows;

    if(!image.data) {
        printf("Error loading image \n");
    }

}

int GreyImage::getPixel(int x, int y){
    return (int)image.at<uchar>(y, x);
}

int GreyImage::getWidth() {
    return width;
}

int GreyImage::getHeight() {
    return height;
}

int* GreyImage::getImage() {
    int size = getHeight() * getWidth();
    int* Image = new int[size];

    for(int i=0; i<getHeight(); i++)
        for (int j=0; j<getWidth(); j++)
            Image[i * image.cols + j] = (int)image.at<uchar>(i, j);

    return Image;
}

void GreyImage::display() {
    imshow("greyImage", image);
}