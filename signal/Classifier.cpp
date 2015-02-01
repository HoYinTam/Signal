#include "Classifier.h"
using namespace std;
using namespace cv;
Classifier::Classifier(){
}
void Classifier::prepare(const vector<Size>&scales){
}
void Classifier::trainFern(const vector<std::pair<std::vector<int>,int> >& fern_data,int resample){
}
void Classifier::trainNN(const vector<cv::Mat>& nn_data){
}
void Classifier::evaluate(const vector<pair<vector<int>,int> >& nNN,const vector<cv::Mat>& nNNT){
}
void Classifier::getFeatures(const cv::Mat& image,const int& scale_idx,std::vector<int>& fern){
}