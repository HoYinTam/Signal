#include "Classifier.h"
using namespace std;
using namespace cv;

Classifier::Classifier(){
	valid = 0.5;
	ncc_thesame = 0.95;
	nstructs = 10;
	structSize = 13;
	thr_fern = 0.6;
	thr_nn = 0.65;
	thr_nn_valid = 0.7;
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
	int leaf;
	for(int t=0;t<nstructs;t++){
		leaf=0;
		for(int f=0;f<structSize;f++){
			leaf = (leaf << 1) + features[scale_idx][t*nstructs+f](image);
		}
		fern[t]=leaf;
	}
}

void Classifier::NNConf(const Mat& example,vector<int>& isin,float& rsconf,float& csconf){
}

float Classifier::measure_forest(vector<int> fern){
	float votes=0;
	for(int i=0;i<nstructs;i++){
		//posteriors[i][idx] = ((float)(pCounter[i][idx]))/(pCounter[i][idx] + nCounter[i][idx]);  
		votes+=posteriors[i][fern[i]];
	}
	return votes;
}

void Classifier::show(){
}