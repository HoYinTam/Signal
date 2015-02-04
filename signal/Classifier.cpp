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
	acum=0;
	int totalFeatures=nstructs*structSize;
	features=vector<vector<Feature>>(scales.size(),vector<Feature>(totalFeatures));
	RNG& rng=theRNG();
	float x1f,x2f,y1f,y2f;
	int x1,x2,y1,y2;
	//slide window feature
	for(int i=0;i<totalFeatures;i++){
		x1f=(float)rng;
		x2f=(float)rng;
		y1f=(float)rng;
		y2f=(float)rng;
		for(int s=0;s<scales.size();s++){
			x1=x1f*scales[s].width;
			x2=x2f*scales[s].width;
			y1=y1f*scales[s].height;
			y2=y2f*scales[s].height;
			features[s][i]=Feature(x1,y1,x2,y2);
		}
	}
	//thresold
	thrN=0.5*nstructs;
	//initalize posteriors
	for(int i=0;i<nstructs;i++){
		posteriors.push_back(vector<float>(pow(2.0,structSize), 0));
		pCounter.push_back(vector<int>(pow(2.0,structSize), 0));
		nCounter.push_back(vector<int>(pow(2.0,structSize), 0));
	}
}

//update pCounter &nCounter,calculate the posteriors
void Classifier::update(const vector<int>& fern, int C, int N){
	int idx;
	for(int i=0;i<nstructs;i++){
		idx=fern[i];
		(C==1)?pCounter[i][idx]+=N:nCounter[i][idx]+=N;
		if(pCounter[i][idx]==0) posteriors[i][idx]=0;
		else posteriors[i][idx] = ((float)(pCounter[i][idx]))/(pCounter[i][idx] + nCounter[i][idx]);
	}
}

void Classifier::trainFern(const vector<std::pair<std::vector<int>,int> >& fern_data,int resample){
	thrP=thr_fern*nstructs;
	for(int i=0;i<fern_data.size();i++){
		if(fern_data[i].second==1){
			if(measure_forest(fern_data[i].first)<=thrP)//occur classifier error
				update(fern_data[i].first,1,1);
		}
		else{
			if(measure_forest(fern_data[i].first)>=thrN)
				update(fern_data[i].first,0,1);
		}
	}
}

void Classifier::trainNN(const vector<cv::Mat>& nn_data){
	float conf,dummy;
	vector<int> y(nn_data.size(),0);
	y[0]=1;//the only one pEx at the first of nn_data;
	vector<int> isin;
	for(int i=0;i<nn_data.size();i++){
		//calculate the similarity between pic and model
		NNConf(nn_data[i],isin,conf,dummy);
		if(y[i]==1&&conf<=thr_nn){//occur classifiy error,update model
			if(isin[1]<0){
				pEx=vector<Mat>(1,nn_data[i]);
				continue;
			}
			pEx.push_back(nn_data[i]);
		}
		if(y[i]==0&&conf>0.5) nEx.push_back(nn_data[i]);
	}
	acum++;
	cout<<acum<<". Trained NN examples: "<<pEx.size()<<" positive "<<nEx.size()<<" negative\n";
}

void Classifier::evaluate(const vector<pair<vector<int>,int> >& nFernT,const vector<cv::Mat>& nNNT){
	float fconf;
	for(int i=0;i<nFernT.size();i++){
		fconf=measure_forest(nFernT[i].first)/nstructs;
		if(fconf>thr_fern) thr_fern=fconf;
	}
	vector<int>isin;
	float conf,dummy;
	for(int i=0;i<nNNT.size();i++){
		NNConf(nNNT[i],isin,conf,dummy);
		if(conf>thr_nn) thr_nn=conf;
	}
	if(thr_nn_valid<thr_nn) thr_nn_valid=thr_nn;
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
	isin=vector<int>(3,-1);
	if(pEx.empty()){
		rsconf=0;
		csconf=0;
		return;
	}
	if(nEx.empty()){
		rsconf=1;
		csconf=1;
		return;
	}
	Mat ncc(1,1,CV_32F);
	float nccP,csmaxP,maxP=0;
	bool anyP=false;
	int maxPidx,validatedPart=ceil(pEx.size()*valid);
	float nccN,maxN=0;
	bool anyN=false;
	//calculate positive example
	for(int i=0;i<pEx.size();i++){
		matchTemplate(pEx[i],example,ncc,CV_TM_CCORR_NORMED);
		nccP=(((float*)ncc.data)[0]+1)*0.5;//calculate  match similarity
		if(nccP>ncc_thesame) anyP=true;
		if(nccP>maxP){
			maxP=nccP;
			maxPidx=i;
			if(i<validatedPart) csmaxP=maxP;
		}
	}
	//calculate negative example
	for(int i=0;i<nEx.size();i++){
		matchTemplate(nEx[i],example,ncc,CV_TM_CCORR_NORMED);
		nccN=(((float*)ncc.data)[0]+1)*0.5;//calculate  match similarity
		if(nccN>ncc_thesame) anyN=true;
		if(nccN>maxN)  maxN=nccN;
	}
	//set isin
	if(anyP) isin[0]=1;
	isin[1]=maxPidx;
	if(anyN) isin[2]=1;
	//measure relative similarity
	float dN=1-maxN,dP=1-maxP;
	rsconf=dN/(dN+dP);
	//Measure Conservative Similarity
	dP = 1 - csmaxP;
	csconf =dN / (dN + dP);
}

float Classifier::measure_forest(vector<int> fern){
	float votes=0;
	for(int i=0;i<nstructs;i++){
		//posteriors[i][idx] = ((float)(pCounter[i][idx]))/(pCounter[i][idx] + nCounter[i][idx]);  
		votes+=posteriors[i][fern[i]];
	}
	return votes;
}

//show postive example
void Classifier::show(){
	Mat examples((int)pEx.size()*pEx[0].rows,pEx[0].cols,CV_8U);
	double minval;
	Mat ex(pEx[0].rows,pEx[0].cols,pEx[0].type());
	for (int i=0;i<pEx.size();i++){
		minMaxLoc(pEx[i],&minval);
		pEx[i].copyTo(ex);
		ex = ex-minval;
		Mat tmp = examples.rowRange(Range(i*pEx[i].rows,(i+1)*pEx[i].rows));
		ex.convertTo(tmp,CV_8U);
	}
	imshow("Examples",examples);
}