#include<opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "Classifier.h";
#include "Tracker.h";
#pragma once
using namespace cv;
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
	public:
		TLD();
};