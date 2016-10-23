//
//  red-pixel-detection.cpp
//  vision1
//
//  Created by Saurabh Pathak on 23/10/2016.
//  Copyright © 2016 Trinity. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

int main2(int argc, const char** argv) {
    
    // reading image in Mat object
    Mat image = imread("img.jpg", 1);
    
    // checking if image was read successfully
    if( image.empty())
    {
        // if image object is empty, display error message on console
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    
    // declaring varibales to store HSV converted image
    // and red pixel upper, lower hue range
    Mat hsv_image, lower_red_hue_range, upper_red_hue_range;
    
    // converting BGR image to HSV
    cvtColor(image, hsv_image, COLOR_BGR2HSV);
    
    // thresholding image as per hue range of red pixels
    // i.e. 0-10 and 160-179. keeping only red pixels
    inRange(hsv_image, Scalar(0, 100, 100), Scalar(10, 255, 255), lower_red_hue_range);
    inRange(hsv_image, Scalar(160, 100, 100), Scalar(179, 255, 255), upper_red_hue_range);
    
    Mat red_hue_image;
    
    // adding weighted images obtained from hue thresholded images
    addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);
    
    
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", red_hue_image );
    
    waitKey(0);
    return 0;
}
