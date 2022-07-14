//
// Created by Nicco on 11/07/2022.
//

#ifndef INTEGRALIMAGESCUDA_GREYIMAGE_H
#define INTEGRALIMAGESCUDA_GREYIMAGE_H

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

class GreyImage {
private:
    int width, heigth;
    Mat image;


public:
    GreyImage(String path);
    int getPixel(int x, int y);
    int getWidth();
    int getHeight();
    int* getImage();
    void display();
};


#endif //INTEGRALIMAGESCUDA_GREYIMAGE_H
