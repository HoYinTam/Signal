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
			generator(img,pt,wraped,hull.size(),rng);
			for(int j=0;j<goodWindow.size();j++){
				idx=goodWindow[j];
				patch=img(grid[idx]);//get the goodWindow infomation
				classifier.getFeatures(patch,grid[idx].scaleIndex,fern);
				pFern.push_back(make_pair(fern,1));
			}
		}
	}
	cout<<"Positive example generated: ferns:"<<pFern.size()<<"  NN:1\n";
}

//normalization
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
	int a=0;
	for(int j=0;j<badWindow.size();j++){
		idx=badWindow[j];
		if (getVar(grid[idx],iisum,iisqsum)<var*1.0f) continue;
		patch=frame(grid[idx]);
		classifier.getFeatures(patch,grid[idx].scaleIndex,fern);
		nFern.push_back(make_pair(fern,0));
		a++;
	}
	cout<<"Negative example generated: ferns:"<<a;
	Scalar dum1,dum2;
	nNN=vector<Mat>(bad_patches);
	for(int j=0;j<bad_patches;j++){
		idx=badWindow[j];
		patch=frame(grid[idx]);
		getPattern(patch,nNN[j],dum1,dum2);
	}
	cout<<"  NN:"<<nNN.size()<<endl;
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

void TLD::track(const Mat& img1,const Mat& img2,vector<Point2f>ps1,vector<Point2f>ps2){
	//generate points
	winPoint(ps1,lastWindow);
	if(ps1.size()<1){
		cout<<"Points didn't generated.\n";
		trackValid=false;
		tracked=false;
		return;
	}
	vector<Point2f> ps=ps1;
	//Frame-to-frame tracking with forward-backward error cheking
	tracked=tracker.trackf2f(img1,img2,ps,ps2);
	if(tracked){
		//window prediction
		winPredict(ps,ps2,lastWindow,trackWindow);
		if(tracker.getFB()>10||trackWindow.x>img2.cols||trackWindow.y>img2.rows||trackWindow.br().x<1||trackWindow.br().y<1){
			//track fail:FB error>10 or trackWindow out of range
			trackValid=false;
			tracked=false;
			cout<<"too unstable prediction FB error="<<tracker.getFB()<<endl;
			return ;
		}
		//estimate validity and confidence
		Mat pattern;
		Scalar mean,stdev;
		window win;
		win.x=max(trackWindow.x,0);
		win.y=max(trackWindow.y,0);
		win.width=min(min(img2.cols-trackWindow.x,trackWindow.width),min(trackWindow.width,trackWindow.br().x));
		win.height=min(min(img2.rows-trackWindow.y,trackWindow.height),min(trackWindow.height,trackWindow.br().y));
		getPattern(img2(win),pattern,mean,stdev);
		vector<int> isin;
		float dummy;
		classifier.NNConf(pattern,isin,dummy,trackConf);
		trackValid=lastValid;
		if(trackConf>classifier.thr_nn_valid) trackValid=true;
	}
	else cout<<"No points track\n";
}

//gernernate 10*10 feature points in slide-window
void TLD::winPoint(vector<Point2f>points,const window& win){
	int max_pt=10;
	int stepx=ceil(win.width/max_pt);
	int stepy=ceil(win.height/max_pt);
	for(int y=win.y;y<win.height+win.y;y+=stepy)
		for(int x=win.x;y<win.width+win.x;x+=stepx)
			points.push_back(Point2f(x,y));
}

void TLD::winPredict(const vector<Point2f> &ps1,const vector<Point2f> &ps2,const window& win1,window& win2){
	vector<float> xoff(ps1.size()),yoff(ps1.size());//point offset
	cout<<"Tracking points:"<<ps1.size()<<endl;
	for(int i=0;i<ps1.size();i++){
		xoff[i]=ps2[i].x-ps1[i].x;
		yoff[i]=ps2[i].y-ps1[i].y;
	}
	//calculate the scale change between 2 windows
	float s,dx=median(xoff),dy=median(yoff);
	if(ps1.size()>1){
		vector<float> d;
		d.reserve(ps1.size()*(ps1.size()-1)/2);
		for(int i=0;i<ps1.size();i++)
			for(int j=i+1;j<ps1.size();j++)
				d.push_back(norm(ps2[i]-ps2[j])/norm(ps1[i]-ps1[j]));
		s=median(d);
	}
	else s=1.0;
	float sx=0.5*(s-1)*win1.width,sy=0.5*(s-1)*win1.height;
	cout<<"s:"<<s<<" "<<"sx:"<<sx<<" "<<"sy:"<<sy<<endl;
	win2.x=round(win1.x+dx-sx);
	win2.y=round(win1.y+dy-sy);
	win2.width=round(win1.width*s);
	win2.height=round(win1.height*s);
	cout<<"predicted window:"<<win2.x<<" "<<win2.y<<" "<<win2.br().x<<" "<<win2.br().y<<endl;
}

void TLD::detect(const Mat& frame){
	//cleaning
	detectWindow.clear();
	detectConf.clear();
	dt.win.clear();

	Mat img(frame.rows,frame.cols,CV_8U);
	integral(frame,iisum,iisqsum);
	GaussianBlur(frame,img,Size(9,9),1.5);
	int numtrees=classifier.getNumStructs();
	float fern_th=classifier.getFernTh();
	vector<int>ferns(10);
	float conf;
	int a=0;
	Mat patch;
	for(int i=0;i<grid.size();i++){
		if(getVar(grid[i],iisum,iisqsum)>=var){//variance detector
			a++;
			//fern detector
			patch=img(grid[i]);
			classifier.getFeatures(patch,grid[i].scaleIndex,ferns);
			conf=classifier.measure_forest(ferns);
			temp.conf[i]=conf;
			temp.pattern[i]=ferns;
			if(conf>numtrees*fern_th) dt.win.push_back(i);//have the target
		}
		else temp.conf[i]=0.0;
	}
	int detections=dt.win.size();
	cout<<a<<" windows passed variance filter\n";
	cout<<detections<<" inital detections from Fern\n";
	if(detections>100){
		nth_element(dt.win.begin(),dt.win.begin()+100,dt.win.end(),CComparator(temp.conf));
		dt.win.resize(100);
		detections=100;
	}
	if(detections==0){
		detected=false;
		return ;
	}
	cout<<"fern detector made "<<detections<<" detections\n";
	//initalize detection structure
	dt.pattern=vector<vector<int>>(detections,vector<int>(10,0));//Corresponding codes of the Ensemble Classifier
	dt.conf1=vector<float>(detections); //Relative Similarity (for final nearest neighbour classifier)
	dt.conf2=vector<float>(detections);//  Conservative Similarity (for integration with tracker)
	dt.isin = vector<vector<int> >(detections,vector<int>(3,-1));        //  Detected (isin=1) or rejected (isin=0) by nearest neighbour classifier
	dt.patch = vector<Mat>(detections,Mat(patch_size,patch_size,CV_32F));//  Corresponding patches
	int idx;
	Scalar mean,stdev;
	float nn_th=classifier.getNNTh();
	//NN detector
	for(int i=0;i<detections;i++){
		idx=dt.win[i];
		patch=frame(grid[idx]);
		getPattern(patch,dt.patch[i],mean,stdev);
		classifier.NNConf(dt.patch[i],dt.isin[i],dt.conf1[i],dt.conf2[i]);
		dt.pattern[i]=temp.pattern[idx];
		if(dt.conf1[i]>nn_th){
			detectWindow.push_back(grid[idx]);
			detectConf.push_back(dt.conf2[i]);
		}
	}
	if(detectWindow.size()>0){
		detected=true;
		cout<<"Found "<<detectWindow.size()<<" NN matches\n";
	}
	else {
		detected=false;
		cout<<"No NN matches found\n";
	}
}

void TLD::learn(const Mat& img){
	window win;
	win.x=max(lastWindow.x,0);
	win.y=max(lastWindow.y,0);
	win.width = min(min(img.cols-lastWindow.x,lastWindow.width),min(lastWindow.width,lastWindow.br().x));
	win.height = min(min(img.rows-lastWindow.y,lastWindow.height),min(lastWindow.height,lastWindow.br().y));
	Scalar mean,stdev;
	Mat pattern;
	getPattern(img(win),pattern,mean,stdev);
	vector<int>isin;
	float dummy,conf;
	classifier.NNConf(pattern,isin,dummy,conf);
	if(conf<0.5){
		cout<<"fast change..not training\n";
		lastValid=false;
		return ;
	}
	if(pow(stdev.val[0],2)<var){
		cout<<"low variance..not training\n";
		lastValid=false;
		return ;
	}
	if(isin[2]==1){
		cout<<"patch in negative data...not training\n";
		lastValid=false;
		return;
	}
	///data generation
	for(int i=0;i<grid.size();i++) grid[i].overlap=winOverlap(lastWindow,grid[i]);
	vector<pair<vector<int>,int>> fern_ex;
	goodWindow.clear();
	badWindow.clear();
	getOverlapWin(lastWindow,num_closet);
	if(goodWindow.size()>0) generatePdata(img,num_wraps);
	else{
		lastValid=false;
		cout<<"No good boxe...not training\n";
		return;
	}
	fern_ex.reserve(pFern.size()+badWindow.size());
	fern_ex.assign(pFern.begin(),pFern.end());
	int idx;
	for(int i=0;i<badWindow.size();i++){
		idx=badWindow[i];
		if(temp.conf[idx]>=1) fern_ex.push_back(make_pair(temp.pattern[idx],0));
	}
	vector<Mat> NN_ex;
	NN_ex.reserve(dt.win.size()+1);
	NN_ex.push_back(pNN);
	for(int i=0;i<dt.win.size();i++){
		idx=dt.win[i];
		if(winOverlap(lastWindow,grid[idx])<bad_overlap) NN_ex.push_back(dt.patch[i]);
	}
	//classifier update
	classifier.trainFern(fern_ex,2);
	classifier.trainNN(NN_ex);
	classifier.show();
}

void TLD::buildgrid(const Mat& img,const Rect& ROI){
	const float SHIFT = 0.1;
	const float SCALES[] = {0.16151,0.19381,0.23257,0.27908,0.33490,0.40188,0.48225,
                          0.57870,0.69444,0.83333,1,1.20000,1.44000,1.72800,
                          2.07360,2.48832,2.98598,3.58318,4.29982,5.15978,6.19174};
	int width, height,min_win_side;
	window win;
	Size scale;
	int sc=0;
	for(int s=0;s<21;s++){
		width=round(ROI.width*SCALES[s]);
		height=round(ROI.height*SCALES[s]);
		min_win_side=min(width,height);
		if(width>img.cols||height>img.rows) continue;
		scale.width=width;
		scale.height=height;
		scales.push_back(scale);
		for(int y=1;y<img.rows-height;y+=round(SHIFT*min_win_side))
			for(int x=1;x<img.cols-width;x+=round(SHIFT*min_win_side)){
				win.x=x;
				win.y=y;
				win.width=width;
				win.height=height;
				win.overlap=winOverlap(win,window(win));
				win.scaleIndex=sc;
				grid.push_back(win);
			}
	sc++;
	}
}

//calculate the overlap percentage
float TLD::winOverlap(const window& box1,const window& box2){
  if (box1.x > box2.x+box2.width) { return 0.0; }
  if (box1.y > box2.y+box2.height) { return 0.0; }
  if (box1.x+box1.width < box2.x) { return 0.0; }
  if (box1.y+box1.height < box2.y) { return 0.0; }

  float colInt =  min(box1.x+box1.width,box2.x+box2.width) - max(box1.x, box2.x);
  float rowInt =  min(box1.y+box1.height,box2.y+box2.height) - max(box1.y,box2.y);

  float intersection = colInt * rowInt;
  float area1 = box1.width*box1.height;
  float area2 = box2.width*box2.height;
  return intersection / (area1 + area2 - intersection);
}

//classify goodWindow and badWindow
void TLD::getOverlapWin(const window& box1,int num_closest){
  float max_overlap = 0;
  for (int i=0;i<grid.size();i++){
      if (grid[i].overlap > max_overlap) {
          max_overlap = grid[i].overlap;
          bestWindow = grid[i];
      }
      if (grid[i].overlap > 0.6){
          goodWindow.push_back(i);
      }
      else if (grid[i].overlap < bad_overlap){
          badWindow.push_back(i);
      }
  }
  //Get the best num_closest (10) boxes and puts them in good_boxes
  if (goodWindow.size()>num_closest){
    nth_element(goodWindow.begin(),goodWindow.begin()+num_closest,goodWindow.end(),OComparator(grid));
    goodWindow.resize(num_closest);
  }
  getHull();
}

//goodWindow's envelope
void TLD::getHull(){
  int x1=INT_MAX, x2=0;
  int y1=INT_MAX, y2=0;
  int idx;
  for (int i=0;i<goodWindow.size();i++){
      idx= goodWindow[i];
      x1=min(grid[idx].x,x1);
      y1=min(grid[idx].y,y1);
      x2=max(grid[idx].x+grid[idx].width,x2);
      y2=max(grid[idx].y+grid[idx].height,y2);
  }
  hull.x = x1;
  hull.y = y1;
  hull.width = x2-x1;
  hull.height = y2 -y1;
}

bool TLD::winCmp(const window& b1,const window& b2){
  TLD t;
    if (t.winOverlap(b1,b2)<0.5)
      return false;
    else
      return true;
}

/*
int  TLD::clusterWin(const vector<window>& detectWindow,vector<int>& index){
	const int c=detectWindow.size();
	//bulid proximity matrix
	Mat D(c,c,CV_32F);
	float d;
	for(int i=0;i<c;i++)
		for(int j=i+1;j<c;j++){
			d=1-winOverlap(detectWindow[i],detectWindow[j]);
			D.at<float>(i,j)=D.at<float>(j,i)=d;
		}
	//initalize disjoint clustering
	int m=c;
	float *L=new float [c];
	int* *nodes=new int* [c];
	int *belongs=new int [c];
	for(int i=0;i<c;i++) {
		belongs[i]=i;
		nodes[i]=new int [2];
	}
	//find nearset neighbor
	for(int it=0;it<c-1;it++){
		float min_d=1;
		int node_a,node_b;
		for(int i=0;i<D.rows;i++)
			for(int j=i+1;j<D.cols;j++){
				if(D.at<float>(i,j)<min_d&&belongs[i]!=belongs[j]){
					min_d=D.at<float>(i,j);
					node_a=i;
					node_b=j;
				}
			}
		if(min_d>0.5){
			int max_idx=0;
			bool vis;
			for(int j=0;j<c;j++){
				vis=false;
				for(int i=0;i<2*c-1;i++){
					if(belongs[j]==i){
						index[j]=max_idx;
						vis=true;
					}
				}
				if(vis) max_idx++;
			}
			return max_idx;
		}
		//Merge clusters and assign level 
		L[m]=min_d;
		nodes[it][0]=belongs[node_a];
		nodes[it][1]=belongs[node_b];
		for(int k=0;k<c;k++){
			if(belongs[k]==belongs[node_a]||belongs[k]==belongs[node_b])
				belongs[k]=m;
		}
		m++;
	}
	delete [] L;
	delete [] belongs;
	for(int i=0;i<c-1;i++) delete [] nodes[i];
	delete [] nodes;
	return 1;
}
*/
void TLD::clusterConf(const vector<window>& detectWindow,const vector<float>& detectConf,vector<window>& clusterWindow,std::vector<float>& cConf){
	int numWin=detectWindow.size();
	vector<int>T;
	float space_thr=0.5;
	int c=1;//the number of class
	switch(numWin){
	case 1:
		clusterWindow=vector<window>(1,detectWindow[0]);
		cConf=vector<float>(1,detectConf[0]);
		return;
		break;
	case 2:
		T=vector<int>(2,0);
		if(1-winOverlap(detectWindow[0],detectWindow[1])>space_thr){
			T[1]=1;
			c=2;
		}
		break;
	default:
		T=vector<int>(numWin,0);
		c=partition(detectWindow,T,winCmp);//divide detectWindow into 2 parts based on overlap>0.5
		break;
	}
	cConf=vector<float>(c);
	clusterWindow=vector<window>(c);
	cout<<"Cluster indexes: ";
	window win;
	for(int i=0;i<c;i++){
		float cnf=0;
		int N=0,mx=0,my=0,mw=0,mh=0;
		for(int j=0;j<T.size();j++){
			if(T[j]==i) {//add up the same kind of window
				cout<<i<<" ";
				cnf+=detectConf[j];
				mx+=detectWindow[j].x;
				my+=detectWindow[j].y;
				mw+=detectWindow[j].width;
				mh+=detectWindow[j].height;
				N++;
			}
		}
		if(N>0){//calculate the representing window
			cConf[i]=cnf/N;
			win.x=round(mx/N);
			win.y=round(my/N);
			win.width=round(mw/N);
			win.height=round(mh/N);
			clusterWindow[i]=win;
		}
	}
	cout<<endl;
}

void TLD::FrameProcess(const Mat& img1,const Mat& img2,vector<Point2f>ps1,vector<Point2f>ps2,window& next,bool& lastWindowfound){
	vector<window> clusterWindow;
	vector<float> cConf;
	int confidentDetections=0;
	int detect_idx;
	///Track
	if(lastWindowfound) track(img1,img2,ps1,ps2);
	else tracked=false;
	///detect
	detect(img2);
	///intergatiom
	if(tracked){
		next=trackWindow;
		lastConf=trackConf;
		lastValid=trackValid;
		cout<<"Tracked\n";
		if(detected){
			clusterConf(detectWindow,detectConf,clusterWindow,cConf);
			cout<<"Found "<<clusterWindow.size()<<" clusters\n";
			for(int i=0;i<clusterWindow.size();i++){
				if(winOverlap(trackWindow,clusterWindow[i])<0.5&&cConf[i]>trackConf){
					confidentDetections++;
					detect_idx=i;
				}
			}
			if(confidentDetections==1){
				cout<<"Found a better match..reinitializing tracking\n";
				next=clusterWindow[detect_idx];
				lastConf=cConf[detect_idx];
				lastValid=false;
			}
			else{
				cout<<confidentDetections<<" confident cluster was found\n";
				int cx=0,cy=0,cw=0,ch=0,close_detections;
				for(int i=0;i<detectWindow.size();i++){
					if(winOverlap(trackWindow,detectWindow[i])>0.7){
						cx+=detectWindow[i].x;
						cy+=detectWindow[i].y;
						cw+=detectWindow[i].width;
						ch+=detectWindow[i].height;
						close_detections++;
						cout<<"weighted detection: "<<detectWindow[i].x<<" "<<detectWindow[i].y
							<<" "<<detectWindow[i].width<<" "<<detectWindow[i].height<<endl;
					}
				}
				if(close_detections>0){
					next.x = cvRound((float)(10*trackWindow.x+cx)/(float)(10+close_detections));   // weighted average trackers trajectory with the close detections
					next.y = cvRound((float)(10*trackWindow.y+cy)/(float)(10+close_detections));
					next.width = cvRound((float)(10*trackWindow.width+cw)/(float)(10+close_detections));
					next.height =  cvRound((float)(10*trackWindow.height+ch)/(float)(10+close_detections));
				}
				else cout<<"0 close detections were found\n";
			}
		}
	}
	else {
		cout<<"No tracking"<<endl;
		lastWindowfound=false;
		lastValid=false;
		if(detected){                           //  and detector is defined
			clusterConf(detectWindow,detectConf,clusterWindow,cConf);   //  cluster detections
			cout<<"Found "<< clusterWindow.size()<<"clusters\n";
          if (cConf.size()==1){
			  next=clusterWindow[0];
              lastConf=cConf[0];
              cout<<"Confident detection..reinitializing tracker\n";
              lastWindowfound = true;
          }
      }
  }
  lastWindow=next;
  if (lastValid) learn(img2);
}