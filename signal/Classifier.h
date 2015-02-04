
#include <opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;
class Classifier{
private:
	int nstructs;
	float thr_fern;
	float thr_nn;
	int structSize;
	float valid;
	float ncc_thesame;
	int acum;
public:
	Classifier();
	float thr_nn_valid;
	void prepare(const vector<Size>&scales);
	void update(const vector<int>& fern, int C, int N);
	void trainFern(const vector<std::pair<std::vector<int>,int> >& fern_data,int resample);
	void trainNN(const vector<cv::Mat>& nn_data);
	void evaluate(const vector<pair<vector<int>,int> >& nFernT,const vector<cv::Mat>& nNNT);
	void getFeatures(const cv::Mat& image,const int& scale_idx,std::vector<int>& fern);
	void NNConf(const Mat& example,vector<int>& isin,float& rsconf,float& csconf);
	float measure_forest(vector<int> fern);
	void show();
	//fern members
	int getNumStructs(){return nstructs;}
	float getFernTh(){return thr_fern;}
	float getNNTh(){return thr_nn;}
	 struct Feature
      {
          uchar x1, y1, x2, y2;
          Feature() : x1(0), y1(0), x2(0), y2(0) {}
          Feature(int _x1, int _y1, int _x2, int _y2)
          : x1((uchar)_x1), y1((uchar)_y1), x2((uchar)_x2), y2((uchar)_y2)
          {}
          bool operator ()(const cv::Mat& patch) const
          { return patch.at<uchar>(y1,x1) > patch.at<uchar>(y2, x2); }
      };
	 vector<vector<Feature> > features;//fern features(one vector for each scale)
	 vector<vector<int>> pCounter;
	 vector<vector<int>> nCounter;
	 vector<vector<float>> posteriors; //ferns posteriors
	 float thrP;
	 float thrN;
	 //NN memebers
	 vector<Mat> pEx;//NN positive example
	 vector<Mat> nEx;
};