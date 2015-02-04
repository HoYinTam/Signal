#include <opencv2/opencv.hpp>
#pragma once
using namespace cv;
using namespace std;

double round(double r);

void drawCircle(Mat& image, vector<Point2f> points,Scalar color);

vector<int> index_shuffle(int begin,int end);
  
float median(vector<float> v);   