#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/photo/photo.hpp"

using namespace cv;

// Creating class for calculating performace measure
class PerformanceMeasure {
double precision, recall, accuracy, specificity, f1;
public:
PerformanceMeasure(double precision, double recall, double accuracy, double specificity, double f1) {
        this->precision = precision;
        this->recall = recall;
        this->accuracy = accuracy;
        this->specificity = specificity;
        this->f1 = f1;
}
void printComparisonResults(string heading) {
        std::cout << heading << "\n";
        std::cout << "Precision is : " << this->precision << "\n";
        std::cout << "Recall is : " << this->recall << "\n";
        std::cout << "Accuracy is : " << this->accuracy << "\n";
        std::cout << "Specificity is : " << this->specificity << "\n";
        std::cout << "f1 is : " << this->f1 << "\n";
}
// Method to compare recognition results
static PerformanceMeasure CompareRecognitionResults( Mat& locations_found, Mat& ground_truth, double& precision, double& recall, double& accuracy, double& specificity, double& f1 )
{
        CV_Assert( locations_found.type() == CV_8UC1 );
        CV_Assert( ground_truth.type() == CV_8UC1 );
        int false_positives = 0;
        int false_negatives = 0;
        int true_positives = 0;
        int true_negatives = 0;
        for (int row=0; row < ground_truth.rows; row++)
                for (int col=0; col < ground_truth.cols; col++)
                {
                        uchar result = locations_found.at<uchar>(row,col);
                        uchar gt = ground_truth.at<uchar>(row,col);
                        if ( gt > 0 )
                                if ( result > 0 )
                                        true_positives++;
                                else false_negatives++;
                        else if ( result > 0 )
                                false_positives++;
                        else true_negatives++;
                }
        precision = ((double) true_positives) / ((double) (true_positives+false_positives));
        recall = ((double) true_positives) / ((double) (true_positives+false_negatives));
        accuracy = ((double) (true_positives+true_negatives)) / ((double) (true_positives+false_positives+true_negatives+false_negatives));
        specificity = ((double) true_negatives) / ((double) (false_positives+true_negatives));
        f1 = 2.0*precision*recall / (precision + recall);

        return *new PerformanceMeasure(precision, recall, accuracy, specificity, f1);
}
};

Mat increaseLuminance(Mat image) {
        Mat luminance_image = Mat::zeros( image.size(), image.type() );
        /// Do the operation new_image(i,j) = alpha*image(i,j) + beta
        for( int y = 0; y < image.rows; y++ )
        { for( int x = 0; x < image.cols; x++ )
          { for( int c = 0; c < 3; c++ )
        {
                luminance_image.at<Vec3b>(y,x)[c] =
                        saturate_cast<uchar>( 2.0*( image.at<Vec3b>(y,x)[c] ) + 0 );
        }}}
        return luminance_image;
}

//----------------------Black/White Pixel Classification----------------------//
Mat getBlackWhitePixels(Mat red_hue_image, Mat image) {
        // smoothing the image
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
                        // drawing child contours
                        if(hierarchy[i][2]<0) {
                                drawContours( drawing, contours, i, Scalar(255,255,255), CV_FILLED, 8, hierarchy, 2 );
                        }
                }
        }
        // doing bitwise AND to obtain target section
        bitwise_and(drawing, image, and_image);
        cvtColor(and_image, and_image, CV_RGB2GRAY);

        // thresholding to obtain clear Black and White sections
        threshold(and_image,and_image, 90, 255, THRESH_BINARY);
        return and_image;
}

//----------------------Red Pixel Detection----------------------//
Mat identifyRedPixels(Mat image) {
        medianBlur(image, image, 5);

        // declaring varibales to store HSV converted image
        // and red pixel upper, lower hue range
        Mat hsv_image, lower_red_hue_range, upper_red_hue_range;

        // converting BGR image to HSV
        cvtColor(image, hsv_image, COLOR_BGR2HSV);
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

        return red_hue_image;
}

int main(int argc, const char** argv) {

        // reading image in Mat object
        Mat image = imread("RoadSignsComposite1.JPG", 1);
        Mat ground_img = imread("RoadSignsCompositeGroundTruth.png", 1);
        Mat out_image;

        // checking if image was read successfully
        if( image.empty())
        {
                // if image object is empty, display error message on console
                std::cout <<  "Image not loaded";
                return -1;
        }

        Mat luminance_image = increaseLuminance(image);

        // Applying methods on real images
        Mat red_luminance_pixels = identifyRedPixels(luminance_image);
        Mat red_pixels = identifyRedPixels(image);
        addWeighted(red_pixels, 1, red_luminance_pixels, 1, 0.0, red_pixels);

        Mat bw_pixels = getBlackWhitePixels(red_pixels, image);

        // Applying methods on ground truth images
        Mat ground_red = identifyRedPixels(ground_img);
        Mat ground_bw = getBlackWhitePixels(ground_red, ground_img);

        imshow( "Black/White Pixel Classification", bw_pixels);
        imshow("Red Pixel Detection", red_pixels);

        imshow( "Black/White Pixel Classification on Ground Truth", ground_bw);
        imshow("Red Pixel Detection on Ground Truth", ground_red);

        //  Comparing results obtained from both real and ground truth images
        double precision, recall, accuracy, specificity, f1;

        // Calculating performance of results obtained with respect to results obtained from
        // ground truth images
        PerformanceMeasure red_measure = PerformanceMeasure::CompareRecognitionResults(red_pixels, ground_red, precision, recall, accuracy, specificity, f1);
        red_measure.printComparisonResults("Red Pixel Result Comparison");

        PerformanceMeasure bw_measure = PerformanceMeasure::CompareRecognitionResults(bw_pixels, ground_bw, precision, recall, accuracy, specificity, f1);
        bw_measure.printComparisonResults("Black/White Result Comparison");

        waitKey(0);
        return 0;
}
