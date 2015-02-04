#include<opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "Classifier.h"
#include "Tracker.h"

using namespace cv;


struct window:public Rect{ //the slide window for operation
	window(){}
	window(Rect r):Rect(r){}
public:
	float overlap; //overlap with current window
	int scaleIndex;
};

struct detection{
	vector<int> win;
	vector<vector<int>> pattern;
	vector<float> conf1;
	vector<float> conf2;
	vector<vector<int>> isin;
	vector<Mat> patch;
};

struct temporal{
	vector<vector<int>> pattern;
	vector<float> conf;
};

struct OComparator{ //overlap comparator
  OComparator(const vector<window>& _grid):grid(_grid){}
  vector<window> grid;
  bool operator()(int idx1,int idx2){
    return grid[idx1].overlap > grid[idx2].overlap;
  }
};

struct CComparator{  //confidence comparator
  CComparator(const vector<float>& _conf):conf(_conf){}
  vector<float> conf;
  bool operator()(int idx1,int idx2){
    return conf[idx1]> conf[idx2];
  }
};


class TLD{
	private:
		PatchGenerator generator;
		Classifier classifier;
		Tracker tracker;
		///parameters
		int step;
		int patch_size;
		//parameters for positive examples
		int num_closet;
		int num_wraps;
		int noise;
		float angle;
		float shift;
		float scale;
		//parameters for negative examples
		float bad_overlap;
		float bad_patches;
		///variable
		//integral image:for the calculation of 2bitBP
		Mat iisum;
		Mat iisqsum;
		float var;
		//training data
		vector<pair<vector<int>,int>> pFern ;//positive ferns<features,labels=1>
		vector<pair<vector<int>,int>> nFern ;//negative ferns<features,labels=1>
		Mat pNN;//positive NearNeighbor classifier example
		vector<Mat> nNN;//negative NN example
		//test data
		vector<pair<vector<int>,int>> nFernT;//negative data to test
		vector<Mat> nNNt;//negative NN examples to test
		//last frame data
		window lastWindow;
		bool lastValid;
		float lastConf;
		//current frame data
		//tracker data
		bool tracked;
		window trackWindow;
		bool trackValid;
		float trackConf;
		//detetor data
		bool detected;
		temporal temp;
		detection dt;
		vector<window> detectWindow;
		vector<bool> detectValid;
		vector<float> detectConf;

		//slide window
		vector<window> grid;
		vector<Size> scales;
		vector<int> goodWindow;//index of slide window with overlap>0.6
		vector<int> badWindow;//index of slide window with overlap<0.2
		window hull; // hull of goodWindow
		window bestWindow; // maximum overlapping window
	public:
		TLD();
		void init(const Mat& frame,const Rect& ROI);
		void generatePdata(const Mat& frame,int num_wraps);
		void generateNdata(const Mat& frame);
		void track(const Mat& img1,const Mat& img2,vector<Point2f>&ps1,vector<Point2f>&ps2);
		void detect(const Mat& frame);
		void clusterConf(const vector<window>& detectWindow,const vector<float>& detectConf,std::vector<window>& clusterWindow,std::vector<float>& cConf);
		void learn(const Mat& img);
		void FrameProcess(const Mat& img1,const Mat& img2,vector<Point2f>&ps1,vector<Point2f>&ps2,window& next,bool& lastWindowfound);
		//useful tool
		void buildgrid(const Mat& img,const Rect& ROI);
		float winOverlap(const window& win1,const window& win2);
		void getOverlapWin(const window& win,int num_closet);
		void getHull();
		void getPattern(const Mat& img,Mat& pattern,Scalar& mean,Scalar& stdev);
		void winPoint(vector<Point2f>&points,const window& win);
		void winPredict(const vector<Point2f> &ps1,const vector<Point2f> &ps2,const window& win1,window& win2);
		double getVar(const window& win,const Mat& sum,const Mat& sqsum);
		//bool winCmp(const window& win1,const window& win2);
};