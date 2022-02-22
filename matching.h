#ifndef MATCHING_H
#define MATCHING_H

#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <dirent.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int baseline_matching(Mat target, char *dir, char *csv_file, vector<string> &top3);
int histogram_matching(Mat target, char *dir, char *csv_file, int bin_num, vector<string> &top3);
int multi_hist_matching(Mat target, char *dir, char *csv_file, int bin_num, vector<string> &top3);
int texture_color_matching(Mat target, char *dir, char *csv_file, int bin_num, vector<string> &top3);
int color_block_seeker(Mat target, char *dir, char *csv_file, int bin_num, vector<string> &top10);
int hsv_hist_matching(Mat target, char *dir, char *csv_file, int bin_num, vector<string> &top3);
int gray_hist_matching(Mat target, char *dir, char *csv_file, int bin_num, vector<string> &top3);
int LOG_matching(Mat target, char *dir, char *csv_file, int bin_num, vector<string> &top3);

#endif
