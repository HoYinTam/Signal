#include"Tracker.h"
#include"utility.h"

using namespace std;
using namespace cv;

Tracker::Tracker(){
	term_criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,20,0.03);
	window_size=Size(4,4);
	level=5;
	lambda=0.5;
}

bool Tracker::trackf2f(const Mat& img1, const Mat& img2,vector<Point2f> &ps1, vector<Point2f> &ps2){
	///OpticalFlowPyrLK
	//forward tracking
	calcOpticalFlowPyrLK(img1,img2,ps1,ps2,status,similarity,window_size,level,term_criteria,lambda,0);
	//backward
	calcOpticalFlowPyrLK(img2,img1,ps2,pointsFB,FB_status,FB_error,window_size,level,term_criteria,lambda,0);
	//compute real FB-error
	for(int i=0;i<ps1.size();i++) FB_error[i]=norm(pointsFB[i]-ps1[i]);
	//Filter out points with FB_error[i] > median(FB_error) && points with sim_error[i] > median(sim_error)
	normCrossCorrelation(img1,img2,ps1,ps2);
	return filterPts(ps1,ps2);
}

void Tracker::normCrossCorrelation(const Mat &img1,const Mat &img2,vector<Point2f>&ps1,vector<Point2f>&ps2){
	Mat rec0(10,10,CV_8U);
	Mat rec1(10,10,CV_8U);
	Mat res(1,1,CV_32F);
	for(int i=0;i<ps1.size();i++){
		if(status[i]==1){//feature points track succeed
			getRectSubPix(img1,Size(10,10),ps1[i],rec0);
			getRectSubPix(img2,Size(10,10),ps2[i],rec1);
			matchTemplate(rec0,rec1,res,CV_TM_CCOEFF_NORMED);
			similarity[i]=((float*)res.data)[0];
		}
		else similarity[i]=0.0;
	}
	rec0.release();
	rec1.release();
	res.release();
}

//find the feature points that FB_error[i] <= median(FB_error) and sim_error[i] > median(sim_error)
bool Tracker::filterPts(vector<Point2f>&ps1,vector<Point2f>&ps2){
	//get error median
	simmed=median(similarity);
	size_t i,k;
	for(i=k=0;i<ps2.size();i++){
		if(!status[i]) continue;
		if(similarity[i]>simmed){//wipe out the points <simmed
			ps1[k]=ps1[i];
			ps2[k]=ps2[i];
			FB_error[k]=FB_error[i];
			k++;
		}
	}
	if(k==0) return false;
	ps1.resize(k);
	ps2.resize(k);
	FB_error.resize(k);

	//get fb_error mediam
	fbmed=median(FB_error);
	for(i=k=0;i<ps2.size();i++){
		if(!status[i]) continue;
		if(FB_error[i] <= fbmed){
			ps1[k]=ps[i];
			ps2[k]=ps[i];
			k++;
		}
	}
	ps1.resize(k);
	ps2.resize(k);
	if(k>0) return true;
	else return false;
}