#include"TLD.h"
#include"utility.h"
using namespace cv;
using namespace std;

TLD::TLD(){
	patch_size=15;
	num_closet=10;
	num_wraps=10;
	noise=5;
	angle=10;
	shift=0.02;
	bad_overlap=0.2;
	bad_patches=100;
	classifier=Classifier();
}

void TLD::init(const Mat& frame,const Rect& ROI){
	//get slide window
	buildgrid(frame,ROI);
	///preparation
	//memory allocation
	iisum.create(frame.row+1,frame.col+1,CV_32F);
	iisqsum.create(frame.row+1,frame.col+1,CV_32F);
	detectConf.reserve(100);
	detectWindow.reserve(100);
	step=7;
	temp.conf.reserve(grid.size());
	temp.pattern.reserve(grid.size());
	dt.bb.reserve(grid.size());
	goodWindow.reserve(grid.size());
	badWindow.reserve(grid.size());
	pNN.create(patch_size,patch_size,CV_64F);
	//Init generator
	generator=PatchGenerator(0,0,5,true,1-0.02,1+0.02,-20/CV_PI*180,20/CV_PI*180,-20/CV_PI*180,20/CV_PI*180);
	getOverlapWin(ROI,10);
	//correct window
	lastWindow=bestWindow;
	lastConf=1;
	lastValid=true;
	///generate positive data
	generatePdata(frame(bestWindow),20);
	//set variance thresold
	Scalar stdev,mean;
	meanStdDev(frame,mean,stdev);
	integral(frame,iisum,iisqsum);
	var=getVar(bestWindow,isum,iisqsum)*0.5;
	//generate negative data
	generateNdata(frame);
	//Split Negative Ferns into Training and Testing sets (they are already shuffled)
	int half=nFern.size ()/2;
	nFernT.assign(nFern.begin()+half,nFern.end());
	nFern.resize(half);
	//Split Negative NN Examples into Training and Testing sets
	half=nNN.size()/2;
	nNNt.assign(nNN.begin()+half,nNN.end());
	nNN.resize(half);
	//merge negative data and postive data and shuffle it
	vector<pair<vector<int>,int>> fern_data(pFern.size()+nFern.size());
	vector<int> index=index_shuffle(0,fern_data.size());
	int i=0;
	for(int j=0;j<pFern.size();j++){
		fern_data[index[i]]=pFern[j];
		i++;
	}
	for(int j=0;j<nFern.size();j++){
		fern_data[index[i]]=nFern[j];
		i++;
	}
	//Data already have been shuffled, just putting it in the same vector
	vector<Mat> NN_data(nNN.size()+1);
	NN_data[0]=pNN;
	i=1;
	for(int j=0;j<nNN.size();j++){
		NN_data[i]=nNN[j];
		i++;
	}
	///training
	classifier.trainFern(fern_data,2);//bootstep=2;
	classifier.trainNN(NN_data);
	///threshold evaluation
	classifier.evaluate(nFernT,nNNt);
}

void TLD::generatePdata(const Mat& frame,int num_wraps){
	Scalar mean,stdev;
	getPattern(frame(bestWindow),pNN,mean,stdev);
	//get fern feature on wraped patches
	Mat img,wraped;
	GaussianBlur(frame,img,Size(9,9),1.5);
	wraped=img(hull);
	RNG &rng=theRNG();
	Point2f pt(hull.x+(hull.width-1)/2,hull.y+(hull.height-1)/2);
	vector<int> fern(classifier.getNumStructs());
	pFern.clear();
	Mat patch;
	if(pFern.capacity()<num_wraps*goodWindow.size())
		pFern.reserve(num_wraps*goodWindow.size());
	int idx;
	for(int i=0;i<num_wraps;i++){
		if(i>0){
			generator(frame,pt,wraped,hull.size(),rng);
			for(int j=0;j<goodWindow.size();j++){
				idx=goodWindow[j];
				patch=img(grid[idx]);//get the goodWindow infomation
				classifier.getFeatures(patch,grid[idx].scaleIndex,fern);
				pFern.push_back(make_pair(fern,1));
			}
		}
	}
}

void TLD::getPattern(const Mat& img,Mat& pattern,Scalar& mean,Scalar& stdev){
	resize(img,pattern,Size(patch_size,patch_size));//resize img to 15*15
	meanStdDev(pattern,mean,stdev);//Computes a mean value and a standard deviation of matrix elements.
	pattern.convertTo(pattern,CV_32F);
	pattern=pattern-mean.val[0];//zero-mean patch
}

void TLD::generateNdata(const Mat& frame){
	random_shuffle(badWindow.begin(),badWindow.end());
	int idx;
	//get Fern Features of the boxes with big variance
	vector<int> fern(classifier.getNumStructs());
	nFern.reserve(badWindow.size());
	Mat patch;
	for(int j=0;j<badWindow.size();j++){
		idx=badWindow[j];
		if (getVar(grid[idx],iisum,iisqsum)<var*0.5f) continue;
		patch=frame(grid[idx]);
		classifier.getFeatures(patch,grid[idx].scaleIndex,fern);
		nFern.push_back(make_pair(fern,0));
	}
	Scalar dum1,dum2;
	nNN=vector<Mat>(bad_patches);
	for(int j=0;j<bad_patches;j++){
		idx=badWindow[i];
		patch=frame(grid[idx]);
		getPattern(patch,nNN[j],dum1,dum2);
	}
}

double TLD::getVar(const window& box,const Mat& sum,const Mat& sqsum){
  double brs = sum.at<int>(box.y+box.height,box.x+box.width);
  double bls = sum.at<int>(box.y+box.height,box.x);
  double trs = sum.at<int>(box.y,box.x+box.width);
  double tls = sum.at<int>(box.y,box.x);
  double brsq = sqsum.at<double>(box.y+box.height,box.x+box.width);
  double blsq = sqsum.at<double>(box.y+box.height,box.x);
  double trsq = sqsum.at<double>(box.y,box.x+box.width);
  double tlsq = sqsum.at<double>(box.y,box.x);
  double mean = (brs+tls-trs-bls)/((double)box.area());
  double sqmean = (brsq+tlsq-trsq-blsq)/((double)box.area());
  return sqmean-mean*mean;
}