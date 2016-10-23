//
//  main.cpp
//  vision1
//
//  Created by Saurabh Pathak on 23/10/2016.
//  Copyright © 2016 Trinity. All rights reserved.
//

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

int main(int argc, const char** argv) {
    
    Mat image = imread("img.jpg", IMREAD_COLOR);
    
    if( image.empty())
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    
    //    int current_threshold = 0, max_threshold = 255;
    
    Mat hsv_image, lower_red_hue_range, upper_red_hue_range;
    
    cvtColor(image, hsv_image, COLOR_BGR2HSV);
    
    inRange(hsv_image, Scalar(0, 100, 100), Scalar(10, 255, 255), lower_red_hue_range);
    inRange(hsv_image, Scalar(160, 100, 100), Scalar(179, 255, 255), upper_red_hue_range);
    
    Mat red_hue_image;
    addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);
    
    //    GaussianBlur(red_hue_image, red_hue_image, Size(9, 9), 2, 2);
    
    //    threshold(hsv_img,otsu_binary_image,current_threshold,max_threshold,
    //              THRESH_BINARY | THRESH_OTSU);
    
    //    Mat otsu_binary_image_display;
    
    //    cvtColor(otsu_binary_image, otsu_binary_image_display, CV_GRAY2BGR);
    
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", red_hue_image );
    
    waitKey(0);
    return 0;
}
