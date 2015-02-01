#include<opencv2/opencv.hpp>
#include<iostream>
#include<cstring>
#include"TLD.h"
#include "utility.h"
using namespace std;
using namespace cv;

//setup for ROI
bool drawingROI=false;//flag for drawing ROI
bool gotROI=false;//flag for the existence of ROI
Rect ROI;

//mouse operation
void on_mouse(int event,int x,int y,int flags,void* param){
	switch (event)
	{
	case CV_EVENT_MOUSEMOVE: 
		{
			if(drawingROI){
				ROI.width=x-ROI.x;
				ROI.height=y-ROI.y;
			}
			break;
		}
	case CV_EVENT_LBUTTONDOWN:
		{
			drawingROI=true;
			ROI=Rect(x,y,0,0);
			break;
		}
	case CV_EVENT_LBUTTONUP:
		{
			drawingROI=false;
			if(ROI.width<0){
				ROI.x+=ROI.width;
				ROI.width=-ROI.width;
			}
			if(ROI.height<0){
				ROI.y+=ROI.height;
				ROI.height=-ROI.height;
			}
			gotROI=true;
			break;
		}
	default:
		{
			cout<<"Warning:Invalid operation\n";
			break;
		}
	}

}

//user interface
bool fromfile=false; //flag for the source of video
bool isRepeat=false; // flag for repeat
void help(){
	cout<<"press \'q\' or Esc to exit\n";
	cout<<"-s    source video\n-q    exit\n";
}
void option(int argc,char *argv[],VideoCapture &capture){
	for(int i=0;i<argc;i++){
		if(strcmp(argv[i],"-s")==0){
			if(i<argc){
				fromfile=true;
				capture.open(argv[i+1]);	
			}
			else help();
		}
	}
}
void UI(int argc,char *argv[],VideoCapture &capture){
	cout<<"***********Tracking-Learing-Detecting************\n";
	cout<<"           made by Ì·ºÆÏÍ ¿ÂÏþºè Íõ½¡ï£            \n";
	cout<<"*************************************************\n";
	help();
	option(argc,argv,capture);
}

int main(int argc,char * argv[]){
	VideoCapture capture;
	capture.open (0);
	UI(argc,argv,capture);

	//init video
	if(!capture.isOpened()){
		cout<<"fail to initialize the video\n";
		return -1;
	}

	///Register the showing window
	cvNamedWindow("TLD",CV_WINDOW_AUTOSIZE);

	///Register mouse callback to draw the ROI
	cvSetMouseCallback("TLD",on_mouse,NULL);

	///initalization
	Mat frame;
	Mat last_gray;
	Mat first;
	capture.set(CV_CAP_PROP_FRAME_WIDTH,340);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT,240);
	while(!gotROI){
		capture>>frame;
		cvtColor(frame,last_gray,CV_RGB2GRAY);
		rectangle(frame,ROI,Scalar(0,0,0),1,8);
		imshow("TLD",frame);
		if(cvWaitKey(33)=='q'||cvWaitKey(33)==27) return 0;
	}
	cout<<"ROI\n"<<"x:"<<ROI.x<<"  "<<"y:"<<ROI.y<<"  "<<"width:"<<ROI.width<<"  "<<"height:"<<ROI.height<<endl;
	cvSetMouseCallback("TLD",NULL,NULL); //remove mouse control
	//TO-DO: TLD initalization
	TLD tld;
	tld.init(last_gray,ROI);
	///Runtime
	Mat now_gray;
	vector<Point2f> pv1,pv2;
	window next;
	bool state=true;
	int frame_count=1,detected_count=1;
	while(capture.read(frame)){
		//modify frame
		cvtColor(frame,now_gray,CV_RGB2GRAY);
		//TO-DO:process
		tld.FrameProcess(last_gray,now_gray,pv1,pv2,next,state);
		//drawing
		if(state){
			drawCircle(frame,pv1,Scalar(255,0,0));
			drawCircle(frame,pv2,Scalar(0,255,0));
			rectangle(frame,pROI,Scalar(0,0,255));
			detected_count++;
		}
		imshow("TLD",frame);
		//ending
		last_gray=now_gray;
		frame_count++;
		pv1.clear();
		pv2.clear();
		cout<<"detection rate:"<<detected_count/frame_count*100<<"%"<<endl;
		if(cvWaitKey(33)=='q'||cvWaitKey(33)==27) return 0;
	}
	waitKey(0);
	return 0;
}
