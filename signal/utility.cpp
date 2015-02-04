#include"utility.h"
using namespace std;
using namespace cv;

double round(double r){
    return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
}

void drawCircle(Mat& image, vector<Point2f> points,Scalar color){
  for( vector<Point2f>::const_iterator i = points.begin(), ie = points.end(); i != ie; ++i )
      {
      Point center( cvRound(i->x ), cvRound(i->y));
      circle(image,*i,2,color,1);
      }
}

vector<int> index_shuffle(int begin,int end){
  vector<int> indexes(end-begin);
  for (int i=begin;i<end;i++){
    indexes[i]=i;
  }
  random_shuffle(indexes.begin(),indexes.end());
  return indexes;
}

float median(vector<float> v){
    int n = floor(v.size() / 2);
    nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}
