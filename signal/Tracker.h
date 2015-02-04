#include <opencv2/opencv.hpp>
#include "utility.h"
using namespace cv;
using namespace std;
class Tracker{
private:
	vector<Point2f> pointsFB;
	Size window_size;//window size of the pyraid
	int level ;//the level of the pyraid
	vector<uchar> status;//the status of the existence of the character 
	vector<uchar> FB_status;
	vector<float> similarity;
	vector<float> FB_error;//forward-backward error
	float simmed;
	float fbmed;
	TermCriteria term_criteria;
	float lambda;
	void normCrossCorrelation(const Mat &img1,const Mat &img2,vector<Point2f>&ps1,vector<Point2f>&ps2);
	bool filterPts(vector<Point2f>&ps1,vector<Point2f>&ps2);
public:
	Tracker();
	bool trackf2f(const Mat& img1, const Mat& img2,vector<Point2f> &points1, vector<Point2f> &points2);
	float getFB(){return fbmed;}
};
