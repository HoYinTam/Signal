#include <opencv2/opencv.hpp>
#pragma once
using namespace cv;
using namespace std;

void drawCircle(Mat& image, vector<Point2f> points,Scalar color){
  for( vector<Point2f>::const_iterator i = points.begin(), ie = points.end(); i != ie; ++i )
      {
      Point center( cvRound(i->x ), cvRound(i->y));
      circle(image,*i,2,color,1);
      }
}