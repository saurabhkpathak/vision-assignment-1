#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/photo/photo.hpp"

using namespace cv;
int main2(int argc, const char** argv) {
    
    // reading image in Mat object
    Mat image = imread("img.jpg", 1);
    
    // checking if image was read successfully
    if( image.empty())
    {
        // if image object is empty, display error message on console
        std::cout <<  "Image not loaded" << std::endl ;
        return -1;
    }
    
    //----------------------Red Pixel Detection----------------------//
    
    // declaring varibales to store HSV converted image
    // and red pixel upper, lower hue range
    Mat hsv_image, lower_red_hue_range, upper_red_hue_range, gray_image;
    
    // converting BGR image to HSV
    cvtColor(image, hsv_image, COLOR_BGR2HSV);
    cvtColor(image, gray_image, CV_RGB2GRAY);
    //imshow("HSV converted image", hsv_image);
    
    // thresholding image as per hue range of red pixels
    // i.e. 0-10 and 160-179. keeping only red pixels
    inRange(hsv_image, Scalar(0, 100, 100), Scalar(10, 255, 255), lower_red_hue_range);
    inRange(hsv_image, Scalar(160, 100, 100), Scalar(179, 255, 255), upper_red_hue_range);
    //imshow("Lower red hue range", lower_red_hue_range);
    //imshow("Upper red hue range", upper_red_hue_range);
    
    Mat red_hue_image;
    
    // adding weighted images obtained from hue thresholded images
    addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);
    //imshow("Red pixel Detection", red_hue_image);
    
    
    
    
    //----------------------Black/White Pixel Classification----------------------//
    
    blur( red_hue_image, red_hue_image, Size(3,3) );
    Mat canny_output, and_image;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    int thresh = 100;
    RNG rng(12345);
    
    /// Detect edges using canny
    Canny( red_hue_image, canny_output, thresh, thresh*2, 3 );
    /// Find contours
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
    
    /// Draw contours
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    if ( !contours.empty() && !hierarchy.empty() ) {
        for( int i = 0; i< contours.size(); i++ )
        {
            if(hierarchy[i][2]<0) {
                drawContours( drawing, contours, i, Scalar(255,255,255), CV_FILLED, 8, hierarchy, 2 );
            }
        }
    }
    
    bitwise_and(drawing, image, and_image);
    cvtColor(and_image, and_image, CV_RGB2GRAY);
    threshold(and_image,and_image, 127, 255, THRESH_BINARY);
    
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", and_image);
    imshow("Original Image", red_hue_image);
    
    waitKey(0);
    return 0;
}
