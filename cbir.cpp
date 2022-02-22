/*
Xichen Liu

CS5330-Project 2

The overall task for this project is, given a database of images and a target image, 
find images in the data with similar content. For this project we will not be using 
neural networks or object recognition methods. Instead, we will focus on more generic 
characteristics of the images such as color, texture, and their spatial layout. This 
will give you practice with working with different color spaces, histograms, spatial 
features, and texture features.
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include "matching.cpp"

#define CVUI_IMPLEMENTATION
#include "cvui.h"

#define WINDOW_NAME "Frame"
#define WINDOW_NAME2 "Target_Image"
#define SHOW_IMAGE "Matching_Result"

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {

    // Mat target = imread("olympus\\pic.0535.jpg");
    // char *dir = (char*)"olympus";
    // char *csv_file = (char*)"data.csv";

    // baseline_matching(target, dir, csv_file);
    // histogram_matching(target, dir, csv_file, 16);
    // multi_hist_matching(target, dir, csv_file, 8);
    // texture_color_matching(target, dir, csv_file, 8);
    // color_block_seeker(target, dir, csv_file, 64);
    // hsv_hist_matching(target, dir, csv_file, 16);
    // gray_hist_matching(target, dir, csv_file, 16);
    // LOG_matching(target, dir, csv_file, 8);

    string target_image = "olympus\\pic.0274.jpg";
    string dir = "olympus";
    string csv_file = "data.csv";
    vector<string> top;

    cout << "Enter path of target image: ";
    cin >> target_image;
    cout << endl;
    cout << "Enter directory name of database: ";
    cin >> dir;
    cout << endl;

    cout << "Press 1 to apply baseline matching" << endl <<
            "Press 2 to apply regular histogram matching" << endl <<
            "Press 3 to apply multiple histogram matching" << endl <<
            "Press 4 to apply color and texture matching" << endl <<
            "Press 5 to apply colored block matching" << endl <<
            "Press 6 to apply hsv histogram matching" << endl <<
            "Press 7 to apply grayscale histogram matching" << endl <<
            "Press 8 to apply laws filter matching" << endl << endl;


    while (true) {

        Mat target = imread(target_image);
        if (target.empty()) {
            cout << "Could not read the image: " << target_image << endl;
            return 1;
        }
        namedWindow("Target Image");
        cvui::init("Target Image");
        imshow("Target Image", target);

        char* dir_name = &dir[0];
        char* csv_name = &csv_file[0];

        char k = waitKey();
        if (k == 'q') {
            exit(-1);
        }
        else if (k == '1') {
            top.clear();
            baseline_matching(target, dir_name, csv_name, top);
            
            Mat top_1 = imread(top[0]);
            imshow("9x9 matches top1", top_1);
            Mat top_2 = imread(top[1]);
            imshow("9x9 matches top2", top_2);
            Mat top_3 = imread(top[2]);
            imshow("9x9 matches top3", top_3);
        }
        else if (k == '2') {
            top.clear();
            histogram_matching(target, dir_name, csv_name, 16, top);
            
            Mat top_1 = imread(top[0]);
            imshow("hist matches top1", top_1);
            Mat top_2 = imread(top[1]);
            imshow("hist matches top2", top_2);
            Mat top_3 = imread(top[2]);
            imshow("hist matches top3", top_3);
        }
        else if (k == '3') {
            top.clear();
            multi_hist_matching(target, dir_name, csv_name, 8, top);
            
            Mat top_1 = imread(top[0]);
            imshow("multi-hist matches top1", top_1);
            Mat top_2 = imread(top[1]);
            imshow("multi-hist matches top2", top_2);
            Mat top_3 = imread(top[2]);
            imshow("multi-hist matches top3", top_3);
        }
        else if (k == '4') {
            top.clear();
            texture_color_matching(target, dir_name, csv_name, 8, top);
            
            Mat top_1 = imread(top[0]);
            imshow("color-texture matches top1", top_1);
            Mat top_2 = imread(top[1]);
            imshow("color-texture matches top2", top_2);
            Mat top_3 = imread(top[2]);
            imshow("color-texture matches top3", top_3);
        }
        else if (k == '5') {
            top.clear();
            color_block_seeker(target, dir_name, csv_name, 8, top);
            
            Mat top_1 = imread(top[0]);
            imshow("middle-color-block matches top1", top_1);
            Mat top_2 = imread(top[1]);
            imshow("middle-color-block matches top2", top_2);
            Mat top_3 = imread(top[2]);
            imshow("middle-color-block matches top3", top_3);
            Mat top_4 = imread(top[3]);
            imshow("middle-color-block matches top4", top_4);
            Mat top_5 = imread(top[4]);
            imshow("middle-color-block matches top5", top_5);
            Mat top_6 = imread(top[5]);
            imshow("middle-color-block matches top6", top_6);
            Mat top_7 = imread(top[6]);
            imshow("middle-color-block matches top7", top_7);
            Mat top_8 = imread(top[7]);
            imshow("middle-color-block matches top8", top_8);
            Mat top_9 = imread(top[8]);
            imshow("middle-color-block matches top9", top_9);
            Mat top_10 = imread(top[9]);
            imshow("middle-color-block matches top10", top_10);
        }
        else if (k == '6') {
            top.clear();
            hsv_hist_matching(target, dir_name, csv_name, 8, top);
            
            Mat top_1 = imread(top[0]);
            imshow("hsv-hist matches top1", top_1);
            Mat top_2 = imread(top[1]);
            imshow("hsv-hist matches top2", top_2);
            Mat top_3 = imread(top[2]);
            imshow("hsv-hist matches top3", top_3);
        }
        else if (k == '7') {
            top.clear();
            gray_hist_matching(target, dir_name, csv_name, 8, top);
            
            Mat top_1 = imread(top[0]);
            imshow("grayscale-hist matches top1", top_1);
            Mat top_2 = imread(top[1]);
            imshow("grayscale-hist matches top2", top_2);
            Mat top_3 = imread(top[2]);
            imshow("grayscale-hist matches top3", top_3);
        }
        else if (k == '8') {
            top.clear();
            LOG_matching(target, dir_name, csv_name, 8, top);
            
            Mat top_1 = imread(top[0]);
            imshow("LOG matches top1", top_1);
            Mat top_2 = imread(top[1]);
            imshow("LOG matches top2", top_2);
            Mat top_3 = imread(top[2]);
            imshow("LOG matches top3", top_3);
        }
        // cvui::checkbox(target, 15, 80, "try it", &check);
        // cvui::update();
    }
    return 0;

}

