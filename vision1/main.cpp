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

int main(int argc, const char * argv[]) {
    VideoCapture cap(0);
    while (true) {
        Mat Webcam;
        cap.read(Webcam);
        imshow("Webcam", Webcam);
    }
    return 0;
}
