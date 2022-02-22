/*
Xichen Liu

CS5330-Project 2

Includes all matching functions used in project
*/

#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <dirent.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/*
reads a string from a CSV file. the 0-terminated string is returned in the char array os.

The function returns false if it is successfully read. It returns true if it reaches the end of the line or the file.
*/
int getstring(FILE *fp, char os[]) {
    int p = 0;
    int eol = 0;
    
    for (;;) {
        char ch = fgetc(fp);
        if (ch == ',') {
            break;
        }
        else if (ch == '\n' || ch == EOF) {
            eol = 1;
            break;
        }
        // printf("%c", ch ); // uncomment for debugging
        os[p] = ch;
        p++;
    }
    
    // printf("\n"); // uncomment for debugging
    os[p] = '\0';

    return eol; // return true if eol
}


int getint(FILE *fp, int *v) {
    char s[256];
    int p = 0;
    int eol = 0;

    for (;;) {
        char ch = fgetc(fp);

        if (ch == ',') {
            break;
        }
        else if (ch == '\n' || ch == EOF) {
            eol = 1;
            break;
        }
        
        s[p] = ch;
        p++;
    }

    s[p] = '\0'; // terminator
    *v = atoi(s);

    return eol; // return true if eol
}


/*
Utility function for reading one float value from a CSV file

The value is stored in the v parameter

The function returns true if it reaches the end of a line or the file
*/
int getfloat(FILE *fp, float *v) {
    char s[256];
    int p = 0;
    int eol = 0;

    for (;;) {
        char ch = fgetc(fp);

        if (ch == ',') {
            break;
        }
        else if (ch == '\n' || ch == EOF) {
            eol = 1;
            break;
        }
        
        s[p] = ch;
        p++;
    }

    s[p] = '\0'; // terminator
    *v = atof(s);

    return eol; // return true if eol
}


/*
comparator to sort the calculated distances
*/
bool cmp(pair<char*, float> &a,pair<char*, float> &b) {
    return a.second < b.second;
}


/*
Function to sort the map according to value in a (key-value) pairs
*/
void sort(map<char*, float> &features, vector<pair<char*, float>> &sorted_distance) { 
    // Copy key-value pair from Map to vector of pairs
    for (auto& it : features) {
        sorted_distance.push_back(it);
    }
  
    // Sort using comparator function
    sort(sorted_distance.begin(), sorted_distance.end(), cmp);

    return;
}


/*
Given a filename, and image filename, and the image features, by
default the function will append a line of data to the CSV format
file.  If reset_file is true, then it will open the file in 'write'
mode and clear the existing contents.

The image filename is written to the first position in the row of
data. The values in image_data are all written to the file as
floats.

The function returns a non-zero value in case of an error.
*/
int append_image_data_csv(char *filename, char *image_filename, vector<float> image_data, int reset_file) {
    char buffer[256];
    char mode[8];
    FILE *fp;

    strcpy(mode, "a");

    if (reset_file) {
        strcpy(mode, "w");
    }
    
    fp = fopen(filename, mode);
    if (!fp) {
        printf("Unable to open output file %s\n", filename);
        exit(-1);
    }
    // write the filename and the feature vector to the CSV file
    strcpy(buffer, image_filename);
    fwrite(buffer, sizeof(char), strlen(buffer), fp);

    for (int i=0; i<image_data.size(); i++) {
        char tmp[256];
        sprintf(tmp, ",%.4f", image_data[i] );
        fwrite(tmp, sizeof(char), strlen(tmp), fp);
    }
        
    fwrite("\n", sizeof(char), 1, fp); // EOL

    fclose(fp);
    
    return 0;
}


/*
Given a file with the format of a string as the first column and
floating point numbers as the remaining columns, this function
returns the filenames as a std::vector of character arrays, and the
remaining data as a 2D std::vector<float>.

filenames will contain all of the image file names.
data will contain the features calculated from each image.

If echo_file is true, it prints out the contents of the file as read
into memory.

The function returns a non-zero value if something goes wrong.
*/
int read_image_data_csv(char *filename, vector<char *> &filenames, vector<vector<float>> &data, int echo_file) {
    FILE *fp;
    float fval;
    char img_file[256];

    fp = fopen(filename, "r");
    if (!fp) {
        printf("Unable to open feature file\n");
        return(-1);
    }

    printf("Reading %s\n", filename);
    for (;;) {
        vector<float> dvec;
        
        // read the filename
        if (getstring(fp, img_file)) {
            break;
        }
        // printf("Evaluting %s\n", filename);

        // read the whole feature file into memory
        for (;;) {
            // get next feature
            float eol = getfloat(fp, &fval);
            dvec.push_back(fval);
            if (eol) break;
        }
        // printf("read %lu features\n", dvec.size());

        data.push_back(dvec);

        char *fname = new char[strlen(img_file) + 1];
        strcpy(fname, img_file);
        filenames.push_back(fname);
    }
    fclose(fp);
    printf("Finished reading CSV file\n\n");

    if (echo_file) {
        for (int i=0; i<data.size(); i++) {
            for (int j=0; j<data[i].size(); j++) {
                printf("%.4f  ", data[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    return 0;
}


/*
Extract features from every image and store the features in a vector
*/
int extract_baseline_features(Mat img, vector<float> &features) {
    
    features.clear();

    int mid_x = img.rows/2;
    int mid_y = img.cols/2;
    for (int i = -4; i <= 4; i++) {
        for (int j = -4; j <= 4; j++) {
            for (int k = 0; k < 3; k++) {
                features.push_back(img.at<Vec3b>(mid_x + i, mid_y + j)[k]);
            }
        }
    }
    
    return 0;
}


/*
Compute histogram of an image
*/
int extract_historgram_rg(Mat img, Mat &hist, int bin_num) {

    float r_val = 0;
    float g_val = 0;
    int r_bin = 0;
    int g_bin = 0;

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            r_val = ((float)img.at<Vec3b>(i, j)[2]) / (img.at<Vec3b>(i, j)[0] + img.at<Vec3b>(i, j)[1] + img.at<Vec3b>(i, j)[2] + 1);
            g_val = ((float)img.at<Vec3b>(i, j)[1]) / (img.at<Vec3b>(i, j)[0] + img.at<Vec3b>(i, j)[1] + img.at<Vec3b>(i, j)[2] + 1);
            r_bin = (int)(r_val * bin_num);
            g_bin = (int)(g_val * bin_num);
            hist.at<int>(r_bin, g_bin)++;
        }
    }
    
    return 0;
}


/*
Compute 2 histograms of an image
*/
int extract_multi_hist(Mat img, Mat &hist_1, Mat &hist_2, int bin_num) {

    int r_bin = 0;
    int g_bin = 0;
    int b_bin = 0;
    int lv = 256/bin_num;

    for (int i = 0; i <= img.rows/2; i++) {
        for (int j = 0; j < img.cols; j++) {
            r_bin = (int)(img.at<Vec3b>(i, j)[2] / lv);
            g_bin = (int)(img.at<Vec3b>(i, j)[1] / lv);
            b_bin = (int)(img.at<Vec3b>(i, j)[0] / lv);
            hist_1.at<int>(r_bin, g_bin, b_bin)++;
        }
    }

    for (int i = img.rows/2 + 1; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            r_bin = (int)(img.at<Vec3b>(i, j)[2] / lv);
            g_bin = (int)(img.at<Vec3b>(i, j)[1] / lv);
            b_bin = (int)(img.at<Vec3b>(i, j)[0] / lv);
            hist_2.at<int>(r_bin, g_bin, b_bin)++;
        }
    }

    return 0;
}


/*
Calculate the magnitude image
*/
int greyscale(Mat &src, Mat &dst) {

    if (!src.data) {
        printf("Error loading src image \n");
        return -1;
    }

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            dst.at<uchar>(i, j) = 0.114 * src.at<Vec3b>(i, j)[0] + 0.587 * src.at<Vec3b>(i, j)[1] + 0.299 * src.at<Vec3b>(i, j)[2];
        }
    }
    return 0;
}


int sobelX3x3(Mat &src, Mat &dst) {

    // generate 1-D sobel filter
    double sobels_x_filter_v[3] = {1, 2, 1};
    double sobels_x_filter_h[3] = {-1, 0, 1};

    for(int i = 0; i < 3; i++) {
        sobels_x_filter_v[i] /= 4;
    }

    // the outer line will not be covered by filter
    int rows = src.rows - 1;
    int cols = src.cols - 1;

    // mid stores the intermidate status after the first separated filter is applied
    Mat mid = dst.clone();

    // go through the pixels with the filter horizontally.
    // since the filter is applied horizontally, we can start from the first row
    for (int i = 0; i < rows; i++) {

        // skip the first line
        for (int j = 1; j < cols; j++) {

            // stores the bgr value
            double color[3] = {0};

            for (int k = -1; k <= 1; k++) {
                Vec3b bgr = src.at<Vec3b>(i, j + k);
                color[0] += sobels_x_filter_h[1 + k] * bgr[0];
                color[1] += sobels_x_filter_h[1 + k] * bgr[1];
                color[2] += sobels_x_filter_h[1 + k] * bgr[2];
            }
            
            Vec3s bgr = {static_cast<short>(color[0]), static_cast<short>(color[1]), static_cast<short>(color[2])};
            mid.at<Vec3s>(i, j) = bgr;
        }
    }

    // go through the pixels with the filter vertically.
    // skip the first line
    for (int i = 1; i < rows; i++) {
        // since the filter is applied horizontally, we can start from the first column
        for (int j = 0; j < cols; j++) {
            // stores the bgr value
            double color[3] = {0};

            for (int k = -1; k <= 1; k++) {
                Vec3b bgr = mid.at<Vec3s>(i + k, j);
                color[0] += sobels_x_filter_v[1 + k] * bgr[0];
                color[1] += sobels_x_filter_v[1 + k] * bgr[1];
                color[2] += sobels_x_filter_v[1 + k] * bgr[2];
            }

            Vec3s bgr = {static_cast<short>(color[0]), static_cast<short>(color[1]), static_cast<short>(color[2])};
            dst.at<Vec3s>(i, j) = bgr;
        }
    }

    return 0;
}


int sobelY3x3(Mat &src, Mat &dst) {

    // generate 1-D sobel filter
    double sobels_y_filter_v[3] = {1, 0, -1};
    double sobels_y_filter_h[3] = {1, 2, 1};

    for(int i = 0; i < 3; i++) {
        sobels_y_filter_h[i] /= 4;
    }

    // the outer line will not be covered by filter
    int rows = src.rows - 1;
    int cols = src.cols - 1;

    Mat mid = dst.clone();

    // go through the pixels with the filter horizontally.
    // since the filter is applied horizontally, we can start from the first row
    for (int i = 0; i < rows; i++) {

        // skip the first line
        for (int j = 1; j < cols; j++) {

            // stores the bgr value
            double color[3] = {0};

            for (int k = -1; k <= 1; k++) {
                Vec3b bgr = src.at<Vec3b>(i, j + k);
                color[0] += sobels_y_filter_h[1 + k] * bgr[0];
                color[1] += sobels_y_filter_h[1 + k] * bgr[1];
                color[2] += sobels_y_filter_h[1 + k] * bgr[2];
            }

            Vec3s bgr = {static_cast<short>(color[0]), static_cast<short>(color[1]), static_cast<short>(color[2])};
            mid.at<Vec3s>(i, j) = bgr;
        }
    }

    // go through the pixels with the filter vertically.
    // skip the first line
    for (int i = 1; i < rows; i++) {

        // since the filter is applied horizontally, we can start from the first column
        for (int j = 0; j < cols; j++) {

            // stores the bgr value
            double color[3] = {0};

            for (int k = -1; k <= 1; k++) {
                Vec3b bgr = mid.at<Vec3s>(i + k, j);
                color[0] += sobels_y_filter_v[1 + k] * bgr[0];
                color[1] += sobels_y_filter_v[1 + k] * bgr[1];
                color[2] += sobels_y_filter_v[1 + k] * bgr[2];
            }
            
            Vec3s bgr = {static_cast<short>(color[0]), static_cast<short>(color[1]), static_cast<short>(color[2])};
            dst.at<Vec3s>(i, j) = bgr;
        }
    }

    return 0;
}


int magnitude(Mat &sx, Mat &sy, Mat &dst) {

    int rows = sx.rows;
    int cols = sx.cols;

    for (int i = 0; i < rows; i++) {

        for (int j = 0; j < cols; j++) {

            double color[3] = {0};

            Vec3s bgr_x = sx.at<Vec3s>(i, j);
            Vec3s bgr_y = sy.at<Vec3s>(i, j);
            color[0] += sqrt((bgr_x[0] * bgr_x[0]) +(bgr_y[0] * bgr_y[0]));
            color[1] += sqrt((bgr_x[1] * bgr_x[1]) +(bgr_y[1] * bgr_y[1]));
            color[2] += sqrt((bgr_x[2] * bgr_x[1]) +(bgr_y[2] * bgr_y[2]));

            Vec3s bgr = {static_cast<short>(color[0]), static_cast<short>(color[1]), static_cast<short>(color[2])};
            dst.at<Vec3s>(i, j) = bgr;
        }
    }

    return 0;
}


int extract_histogram_rgb(Mat img, Mat &hist, int bin_num, vector<int> &texture_feature) {

    int bin = 0;
    int r_bin = 0;
    int g_bin = 0;
    int b_bin = 0;
    int lv = 256/bin_num;

    if (img.channels() == 1) {
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                bin = (int)((int)img.at<uchar>(i, j) / lv);
                texture_feature[bin] += 1;
            }
        }
    }
    else {
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                r_bin = (int)(img.at<Vec3b>(i, j)[2] / lv);
                g_bin = (int)(img.at<Vec3b>(i, j)[1] / lv);
                b_bin = (int)(img.at<Vec3b>(i, j)[0] / lv);
                hist.at<int>(r_bin, g_bin, b_bin)++;
            }
        }
    }
    
    return 0;
}


/*
Apply weighted spatial filter to compute histogram
*/
int middle_blocker_seeker(Mat img, Mat &hist, int bin_num) {

    int bin = 0;
    int r_bin = 0;
    int g_bin = 0;
    int b_bin = 0;
    int lv = 256/bin_num;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            r_bin = (int)(img.at<Vec3b>(i, j)[2] / lv);
            g_bin = (int)(img.at<Vec3b>(i, j)[1] / lv);
            b_bin = (int)(img.at<Vec3b>(i, j)[0] / lv);
            // hist.at<int>(r_bin, g_bin, b_bin)++;
            if (i > (3*(img.rows/7)) && i < (4*(img.rows/7))) {
                if (j > (3*(img.cols/7)) && j < (4*(img.cols/7))) {
                    hist.at<int>(r_bin, g_bin, b_bin) += 30;
                }
                else if ((j > (2*(img.cols/7)) && j < (3*(img.cols/7))) || (j > (4*(img.cols/7)) && j < (5*(img.cols/7)))) {
                    hist.at<int>(r_bin, g_bin, b_bin) += 20;
                }
                else if ((j > (img.cols/7) && j < (2*(img.cols/7))) || (j > (5*(img.cols/7)) && j < (6*(img.cols/7)))) {
                    hist.at<int>(r_bin, g_bin, b_bin) += 2;
                }
                else {
                    hist.at<int>(r_bin, g_bin, b_bin) += 1;
                }
            }
            else if ((i > (2*(img.rows/7)) && i < (3*(img.rows/7))) || (i > (4*(img.rows/7)) && i < (5*(img.rows/7)))) {
                if (j > (3*(img.cols/7)) && j < (4*(img.cols/7))) {
                    hist.at<int>(r_bin, g_bin, b_bin) += 30;
                }
                else if ((j > (2*(img.cols/7)) && j < (3*(img.cols/7))) || (j > (4*(img.cols/7)) && j < (5*(img.cols/7)))) {
                    hist.at<int>(r_bin, g_bin, b_bin) += 20;
                }
                else if ((j > (img.cols/7) && j < (2*(img.cols/7))) || (j > (5*(img.cols/7)) && j < (6*(img.cols/7)))) {
                    hist.at<int>(r_bin, g_bin, b_bin) += 2;
                }
                else {
                    hist.at<int>(r_bin, g_bin, b_bin) += 1;
                }
            }
            else if ((i > (img.rows/7) && i < (2*(img.rows/7))) || (i > (5*(img.rows/7)) && i < (6*(img.rows/7)))) {
                if (j > (3*(img.cols/7)) && j < (4*(img.cols/7))) {
                    hist.at<int>(r_bin, g_bin, b_bin) += 30;
                }
                else if ((j > (2*(img.cols/7)) && j < (3*(img.cols/7))) || (j > (4*(img.cols/7)) && j < (5*(img.cols/7)))) {
                    hist.at<int>(r_bin, g_bin, b_bin) += 20;
                }
                else if ((j > (img.cols/7) && j < (2*(img.cols/7))) || (j > (5*(img.cols/7)) && j < (6*(img.cols/7)))) {
                    hist.at<int>(r_bin, g_bin, b_bin) += 2;
                }
                else {
                    hist.at<int>(r_bin, g_bin, b_bin) += 1;
                }
            }
            else {
                if (j > (3*(img.cols/7)) && j < (4*(img.cols/7))) {
                    hist.at<int>(r_bin, g_bin, b_bin) += 30;
                }
                else if ((j > (2*(img.cols/7)) && j < (3*(img.cols/7))) || (j > (4*(img.cols/7)) && j < (5*(img.cols/7)))) {
                    hist.at<int>(r_bin, g_bin, b_bin) += 20;
                }
                else if ((j > (img.cols/7) && j < (2*(img.cols/7))) || (j > (5*(img.cols/7)) && j < (6*(img.cols/7)))) {
                    hist.at<int>(r_bin, g_bin, b_bin) += 2;
                }
                else {
                    hist.at<int>(r_bin, g_bin, b_bin) += 1;
                }
            }
        }
    }
    return 0;
}


int extract_hsv(Mat img, Mat &hist, int bin_num) {
    int bin = 0;
    int h_bin = 0;
    int s_bin = 0;
    int v_bin = 0;
    int lv = 256/bin_num;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            h_bin = (int)(img.at<Vec3b>(i, j)[0] / lv);
            s_bin = (int)(img.at<Vec3b>(i, j)[1] / lv);
            v_bin = (int)(img.at<Vec3b>(i, j)[2] / lv);
            hist.at<int>(h_bin, s_bin, v_bin)++;
        }
    }
    
    return 0;
}


int extract_grayscale(Mat img, int bin_num, vector<int> &texture_feature) {
    int bin = 0;
    int lv = 256/bin_num;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            bin = (int)((int)img.at<uchar>(i, j) / lv);
            texture_feature[bin] += 1;
        }
    }
    return 0;
}


/*
Extract features from the image set and expend them to csv file
*/
int append_all(char *dir, char *csv_file, int flag, int bin_num) {
    char dirname[256];
    char buffer[256];
    FILE *fp;
    struct dirent *dp;
    DIR *dirp;
    vector<float> features;

    Mat hist;
    Mat hist_2;
    int dim2d[2] = {bin_num, bin_num};
    int dim3d[3] = {bin_num, bin_num, bin_num};

    // earse all the contents in the csv file
    fp = fopen(csv_file, "w");
    fclose(fp);

    // get the directory path
    strcpy(dirname, dir);
    printf("Processing directory: %s\n\n", dirname);

    // open the directory
    dirp = opendir(dirname);
    if (dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }
    

    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL) {
        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif")) {

            // printf("processing image file: %s\n", dp->d_name);

            // build the overall filename
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            // printf("full path name: %s\n", buffer);
            Mat curr_img = imread(buffer);
            if (flag == 1) {
                features.clear();
                extract_baseline_features(curr_img, features);

                // append image name and features into the csv file
                append_image_data_csv(csv_file, buffer, features, false);
            }
            else if (flag == 2) {
                features.clear();
                hist = Mat::zeros(2, dim2d, CV_32S);
                extract_historgram_rg(curr_img, hist, bin_num);
                for (int i = 0; i < hist.rows; i++) {
                    for (int j = 0; j < hist.cols; j++) {
                        features.push_back(hist.at<int>(i, j));
                    }
                }
                append_image_data_csv(csv_file, buffer, features, false);
            }
            else if (flag == 3) {
                features.clear();
                hist = Mat::zeros(3, dim3d, CV_32S);
                hist_2 = Mat::zeros(3, dim3d, CV_32S);
                extract_multi_hist(curr_img, hist, hist_2, bin_num);
                for (int i = 0; i < bin_num; i++) {
                    for (int j = 0; j < bin_num; j++) {
                        for (int k = 0; k < bin_num; k++) {
                            features.push_back(hist.at<int>(i, j, k));
                        }
                    }
                }
                for (int i = 0; i < bin_num; i++) {
                    for (int j = 0; j < bin_num; j++) {
                        for (int k = 0; k < bin_num; k++) {
                            features.push_back(hist_2.at<int>(i, j, k));
                        }
                    }
                }
                append_image_data_csv(csv_file, buffer, features, false);
            }
            else if (flag == 4) {
                vector<int> mag_feature(bin_num);
                features.clear();
                hist = Mat::zeros(3, dim3d, CV_32S);
                extract_histogram_rgb(curr_img, hist, bin_num, mag_feature);
                Mat sobelx_pic = Mat::zeros(curr_img.rows, curr_img.cols, CV_16SC3);
                sobelX3x3(curr_img, sobelx_pic);
                Mat sobely_pic = Mat::zeros(curr_img.rows, curr_img.cols, CV_16SC3);
                sobelY3x3(curr_img, sobely_pic);
                Mat magnitude_pic = Mat::zeros(curr_img.rows, curr_img.cols, CV_16SC3);
                magnitude(sobelx_pic, sobely_pic, magnitude_pic);
                convertScaleAbs(magnitude_pic, magnitude_pic);
                Mat grayscale_pic = Mat::zeros(curr_img.rows, curr_img.cols, CV_8UC1);
                greyscale(magnitude_pic, grayscale_pic);
                extract_histogram_rgb(grayscale_pic, hist, bin_num, mag_feature);


                for (int i = 0; i < bin_num; i++) {
                    for (int j = 0; j < bin_num; j++) {
                        for (int k = 0; k < bin_num; k++) {
                            features.push_back(hist.at<int>(i, j, k));
                        }
                    }
                }
                for (int i = 0; i < bin_num; i++) {
                    features.push_back(mag_feature[i]);
                }
                append_image_data_csv(csv_file, buffer, features, false);
            }
            else if (flag == 5) {
                features.clear();
                hist = Mat::zeros(3, dim3d, CV_32S);
                middle_blocker_seeker(curr_img, hist, bin_num);


                for (int i = 0; i < bin_num; i++) {
                    for (int j = 0; j < bin_num; j++) {
                        for (int k = 0; k < bin_num; k++) {
                            features.push_back(hist.at<int>(i, j, k));
                        }
                    }
                }
                append_image_data_csv(csv_file, buffer, features, false);
            }
            else if (flag == 6) {
                features.clear();
                hist = Mat::zeros(3, dim3d, CV_32S);
                Mat hsv_img;
                cvtColor(curr_img, hsv_img, COLOR_BGR2HSV);
                extract_hsv(hsv_img, hist, bin_num);
                for (int i = 0; i < bin_num; i++) {
                    for (int j = 0; j < bin_num; j++) {
                        for (int k = 0; k < bin_num; k++) {
                            features.push_back(hist.at<int>(i, j, k));
                        }
                    }
                }
                append_image_data_csv(csv_file, buffer, features, false);
            }
            else if (flag == 7) {
                vector<int> mag_feature(bin_num);
                features.clear();
                hist = Mat::zeros(3, dim3d, CV_32S);
                extract_grayscale(curr_img, bin_num, mag_feature);
                for (int i = 0; i < bin_num; i++) {
                    features.push_back(mag_feature[i]);
                }
                append_image_data_csv(csv_file, buffer, features, false);
            }
            else if (flag == 8) {
                Mat log_ker = (Mat_<char>(5, 5) << 1, 0, -2, 0, -1,
                                        0, 0, 0, 0, 0, 
                                        -2, 0, 4, 0, -2, 
                                        0, 0, 0, 0, 0, 
                                        1, 0, -2, 0, 1);

                Mat filtered_img;
                filter2D(curr_img, filtered_img, curr_img.depth(), log_ker);
                vector<int> mag_feature(bin_num);
                features.clear();
                hist = Mat::zeros(3, dim3d, CV_32S);
                extract_histogram_rgb(filtered_img, hist, bin_num, mag_feature);
                for (int i = 0; i < bin_num; i++) {
                    for (int j = 0; j < bin_num; j++) {
                        for (int k = 0; k < bin_num; k++) {
                            features.push_back(hist.at<int>(i, j, k));
                        }
                    }
                }
                append_image_data_csv(csv_file, buffer, features, false);
            }
        }
    }
    
    return 0;
}


/*
Use the 9x9 square in the middle of the image as a feature vector. Use sum-of-squared-difference as the distance metric.
*/
int baseline_matching(Mat target, char *dir, char *csv_file, vector<string> &top3) {

    vector<float> target_feature;       // features of target image
    vector<char*> db_names;             // names of images in the database
    vector<vector<float>> db_fatures;   // features of all images in the database
    float distance;                     // distance between selected image from db and target
    map<char*, float> distance_map;     // dictionary that stores distances between images and target
    vector<pair<char*, float>> sorted_distance_map;

    // extract features from target image
    int mid_x = target.rows/2;
    int mid_y = target.cols/2;
    for (int i = -4; i <= 4; i++) {
        for (int j = -4; j <= 4; j++) {
            for (int k = 0; k < 3; k++) {
                target_feature.push_back(target.at<Vec3b>(mid_x + i, mid_y + j)[k]);
            }
        }
    }
    
    // store all features and read them out
    append_all(dir, csv_file, 1, 0);
    read_image_data_csv(csv_file, db_names, db_fatures, false);

    // calculate all distances and stores them into distance_map
    for (int img_idx = 0; img_idx < db_fatures.size(); img_idx++) {
        distance = 0;
        for (int feature_idx = 0; feature_idx < db_fatures[img_idx].size(); feature_idx++) {
            distance += ((db_fatures[img_idx][feature_idx] - target_feature[feature_idx]) * 
                            (db_fatures[img_idx][feature_idx] - target_feature[feature_idx]));
        }
        distance_map.insert(pair<char*, float>(db_names[img_idx], distance));
    }

    // sort the map and output the top 3 matches
    sort(distance_map, sorted_distance_map);
    printf("Task 1: according to baseline matching, the top three matches to %s are:\n", sorted_distance_map[0].first);
    for (int i = 1; i <= 3; i++) {
        printf(sorted_distance_map[i].first);
        string tmp(sorted_distance_map[i].first);
        top3.push_back(tmp);
        printf("\n");
    }
    printf("\n");

    return 0;
}


/*
Use a single normalized color histogram of your choice (has to be at least two-dimensional) 
as the feature vector. Use histogram intersection as the distance metric. Write your own 
code to calculate the histograms from the image. You can use the cv::Mat to hold the 2-D 
or 3-D histogram data.
*/
int histogram_matching(Mat target, char *dir, char *csv_file, int bin_num, vector<string> &top3) {

    Mat target_hist;
    vector<float> target_feature;
    vector<char*> db_names;             // names of images in the database
    vector<vector<float>> db_fatures;   // features of all images in the database
    float distance;                     // distance between selected image from db and target
    map<char*, float> distance_map;     // dictionary that stores distances between images and target
    vector<pair<char*, float>> sorted_distance_map;
    
    int dim2d[2] = {bin_num, bin_num};

    // extract features from target image
    target_hist = Mat::zeros(2, dim2d, CV_32S);
    
    extract_historgram_rg(target, target_hist, bin_num);
    
    for (int i = 0; i < target_hist.rows; i++) {
        for (int j = 0; j < target_hist.cols; j++) {
            target_feature.push_back(target_hist.at<int>(i, j));
        }
    }
    
    // store all features and read them out
    append_all(dir, csv_file, 2, bin_num);
    read_image_data_csv(csv_file, db_names, db_fatures, false);

    // calculate all distances and stores them into distance_map
    for (int img_idx = 0; img_idx < db_fatures.size(); img_idx++) {
        distance = 0;
        for (int feature_idx = 0; feature_idx < db_fatures[img_idx].size(); feature_idx++) {
            distance += target_feature[feature_idx] - min(db_fatures[img_idx][feature_idx], target_feature[feature_idx]);
        }
        distance_map.insert(pair<char*, float>(db_names[img_idx], distance));
    }

    // sort the map and output the top 3 matches
    sort(distance_map, sorted_distance_map);
    printf("Task 2: according to histogram matching, the top three matches to %s are:\n", sorted_distance_map[0].first);
    for (int i = 1; i <= 3; i++) {
        printf(sorted_distance_map[i].first);
        string tmp(sorted_distance_map[i].first);
        top3.push_back(tmp);
        printf("\n");
    }
    printf("\n");
    return 0;
}


/*
Use two or more color histograms of your choice as the feature vector. The histograms should represent different spatial parts of the image. The parts can be overlapping or disjoint.
*/
int multi_hist_matching(Mat target, char *dir, char *csv_file, int bin_num, vector<string> &top3) {

    Mat target_hist_1;
    Mat target_hist_2;
    vector<float> target_feature;
    vector<char*> db_names;             // names of images in the database
    vector<vector<float>> db_fatures;   // features of all images in the database
    float distance;                     // distance between selected image from db and target
    map<char*, float> distance_map;     // dictionary that stores distances between images and target
    vector<pair<char*, float>> sorted_distance_map;
    
    int dim3d[3] = {bin_num, bin_num, bin_num};

    // extract features from target image
    target_hist_1 = Mat::zeros(3, dim3d, CV_32S);
    target_hist_2 = Mat::zeros(3, dim3d, CV_32S);
    
    extract_multi_hist(target, target_hist_1, target_hist_2, bin_num);
    
    for (int i = 0; i < bin_num; i++) {
        for (int j = 0; j < bin_num; j++) {
            for (int k = 0; k < bin_num; k++) {
                target_feature.push_back(target_hist_1.at<int>(i, j, k));
            }
        }
    }
    for (int i = 0; i < bin_num; i++) {
        for (int j = 0; j < bin_num; j++) {
            for (int k = 0; k < bin_num; k++) {
                target_feature.push_back(target_hist_2.at<int>(i, j, k));
            }
        }
    }
    
    // store all features and read them out
    append_all(dir, csv_file, 3, bin_num);
    read_image_data_csv(csv_file, db_names, db_fatures, false);

    // calculate all distances and stores them into distance_map
    for (int img_idx = 0; img_idx < db_fatures.size(); img_idx++) {
        distance = 0;
        for (int feature_idx = 0; feature_idx < db_fatures[img_idx].size(); feature_idx++) {
            distance += target_feature[feature_idx] - min(db_fatures[img_idx][feature_idx], target_feature[feature_idx]);
        }
        distance_map.insert(pair<char*, float>(db_names[img_idx], distance));
    }

    // sort the map and output the top 3 matches
    sort(distance_map, sorted_distance_map);
    printf("Task 3: according to multi histogram matching, the top three matches to %s are:\n", sorted_distance_map[0].first);
    for (int i = 1; i <= 3; i++) {
        printf(sorted_distance_map[i].first);
        string tmp(sorted_distance_map[i].first);
        top3.push_back(tmp);
        printf("\n");
    }
    printf("\n");
    return 0;
}


/*
Use a whole image color histogram and a whole image texture histogram as the feature vector.
*/
int texture_color_matching(Mat target, char *dir, char *csv_file, int bin_num, vector<string> &top3) {

    Mat target_hist_1;
    vector<int> target_fea_2(bin_num);
    vector<float> target_feature;
    vector<char*> db_names;             // names of images in the database
    vector<vector<float>> db_fatures;   // features of all images in the database
    float distance;                     // distance between selected image from db and target
    float rgb_dis;
    float m_dis;
    map<char*, float> distance_map;     // dictionary that stores distances between images and target
    vector<pair<char*, float>> sorted_distance_map;
    
    int dim3d[3] = {bin_num, bin_num, bin_num};

    // extract features from target image
    target_hist_1 = Mat::zeros(3, dim3d, CV_32S);
    
    extract_histogram_rgb(target, target_hist_1, bin_num, target_fea_2);
    Mat sobelx_pic = Mat::zeros(target.rows, target.cols, CV_16SC3);
    sobelX3x3(target, sobelx_pic);
    Mat sobely_pic = Mat::zeros(target.rows, target.cols, CV_16SC3);
    sobelY3x3(target, sobely_pic);
    Mat magnitude_pic = Mat::zeros(target.rows, target.cols, CV_16SC3);
    magnitude(sobelx_pic, sobely_pic, magnitude_pic);
    convertScaleAbs(magnitude_pic, magnitude_pic);
    Mat grayscale_pic = Mat::zeros(target.rows, target.cols, CV_8UC1);
    greyscale(magnitude_pic, grayscale_pic);
    extract_histogram_rgb(grayscale_pic, target_hist_1, bin_num, target_fea_2);
    
    for (int i = 0; i < bin_num; i++) {
        for (int j = 0; j < bin_num; j++) {
            for (int k = 0; k < bin_num; k++) {
                target_feature.push_back(target_hist_1.at<int>(i, j, k));
            }
        }
    }
    for (int i = 0; i < bin_num; i++) {
        target_feature.push_back(target_fea_2[i]);
    }
    // store all features and read them out
    append_all(dir, csv_file, 4, bin_num);
    read_image_data_csv(csv_file, db_names, db_fatures, false);

    // calculate all distances and stores them into distance_map
    for (int img_idx = 0; img_idx < db_fatures.size(); img_idx++) {
        distance = 0;
        rgb_dis = 0;
        m_dis = 0;
        for (int feature_idx = 0; feature_idx < bin_num*bin_num*bin_num; feature_idx++) {
            rgb_dis += abs(target_feature[feature_idx] - min(db_fatures[img_idx][feature_idx], target_feature[feature_idx]));
        }
        for (int feature_idx = bin_num*bin_num*bin_num; feature_idx < db_fatures[img_idx].size(); feature_idx++) {
            m_dis += abs(target_feature[feature_idx] - min(db_fatures[img_idx][feature_idx], target_feature[feature_idx]));
        }
        distance = 0.25*rgb_dis + 0.75*m_dis;
        distance_map.insert(pair<char*, float>(db_names[img_idx], distance));
    }

    // sort the map and output the top 3 matches
    sort(distance_map, sorted_distance_map);
    printf("Task 4: according to color and texture matching, the top three matches to %s are:\n", sorted_distance_map[0].first);
    for (int i = 1; i <= 3; i++) {
        printf(sorted_distance_map[i].first);        
        string tmp(sorted_distance_map[i].first);
        top3.push_back(tmp);
        printf("\n");
        // cout << sorted_distance_map[i].first << " " << sorted_distance_map[i].second << endl;
    }
    printf("\n");
    return 0;
}


/*
Looking for the images that contain a color block at the middle
*/
int color_block_seeker(Mat target, char *dir, char *csv_file, int bin_num, vector<string> &top10) {

    Mat target_hist;
    // vector<int> target_fea(bin_num);
    vector<float> target_feature;
    vector<char*> db_names;             // names of images in the database
    vector<vector<float>> db_fatures;   // features of all images in the database
    float distance;                     // distance between selected image from db and target
    map<char*, float> distance_map;     // dictionary that stores distances between images and target
    vector<pair<char*, float>> sorted_distance_map;
    
    int dim3d[3] = {bin_num, bin_num, bin_num};

    // extract features from target image
    target_hist = Mat::zeros(3, dim3d, CV_32S);
    
    middle_blocker_seeker(target, target_hist, bin_num);
    
    for (int i = 0; i < bin_num; i++) {
        for (int j = 0; j < bin_num; j++) {
            for (int k = 0; k < bin_num; k++) {
                target_feature.push_back((float)target_hist.at<int>(i, j, k)/10);
            }
        }
    }
    // store all features and read them out
    append_all(dir, csv_file, 5, bin_num);
    read_image_data_csv(csv_file, db_names, db_fatures, false);

    // calculate all distances and stores them into distance_map
    for (int img_idx = 0; img_idx < db_fatures.size(); img_idx++) {
        distance = 0;
        for (int feature_idx = 0; feature_idx < db_fatures[img_idx].size(); feature_idx++) {
            distance += (target_feature[feature_idx] - min(target_feature[feature_idx], db_fatures[img_idx][feature_idx]));
        }
        distance_map.insert(pair<char*, float>(db_names[img_idx], distance));
    }

    // sort the map and output the top 3 matches
    sort(distance_map, sorted_distance_map);
    printf("Task 5: find the top three matches that contains a colored block at the middle to %s are:\n", sorted_distance_map[0].first);
    for (int i = 1; i <= 10; i++) {
        printf(sorted_distance_map[i].first);
        string tmp(sorted_distance_map[i].first);
        top10.push_back(tmp);
        printf("\n");
        // cout << sorted_distance_map[i].first << " " << sorted_distance_map[i].second << endl;
    }
    printf("\n");
    return 0;
}


int hsv_hist_matching(Mat target, char *dir, char *csv_file, int bin_num, vector<string> &top3) {

    Mat target_hist;
    Mat hsv_target;
    vector<float> target_feature;
    vector<char*> db_names;             // names of images in the database
    vector<vector<float>> db_fatures;   // features of all images in the database
    float distance;                     // distance between selected image from db and target
    map<char*, float> distance_map;     // dictionary that stores distances between images and target
    vector<pair<char*, float>> sorted_distance_map;
    
    int dim3d[3] = {bin_num, bin_num, bin_num};

    // extract features from target image
    target_hist = Mat::zeros(3, dim3d, CV_32S);
    cvtColor(target, hsv_target, COLOR_BGR2HSV);

    extract_hsv(hsv_target, target_hist, bin_num);
    
    for (int i = 0; i < bin_num; i++) {
        for (int j = 0; j < bin_num; j++) {
            for (int k = 0; k < bin_num; k++) {
                target_feature.push_back(target_hist.at<int>(i, j, k));
            }
        }
    }
    
    // store all features and read them out
    append_all(dir, csv_file, 6, bin_num);
    read_image_data_csv(csv_file, db_names, db_fatures, false);
    // calculate all distances and stores them into distance_map
    for (int img_idx = 0; img_idx < db_fatures.size(); img_idx++) {
        distance = 0;
        for (int feature_idx = 0; feature_idx < db_fatures[img_idx].size(); feature_idx++) {
            distance += target_feature[feature_idx] - min(db_fatures[img_idx][feature_idx], target_feature[feature_idx]);
        }
        distance_map.insert(pair<char*, float>(db_names[img_idx], distance));
    }

    // sort the map and output the top 3 matches
    sort(distance_map, sorted_distance_map);
    printf("Extension: according to hsv histogram matching, the top three matches to %s are:\n", sorted_distance_map[0].first);
    for (int i = 1; i <= 3; i++) {
        printf(sorted_distance_map[i].first);
        string tmp(sorted_distance_map[i].first);
        top3.push_back(tmp);
        printf("\n");
    }
    printf("\n");
    return 0;
}


int gray_hist_matching(Mat target, char *dir, char *csv_file, int bin_num, vector<string> &top3) {

    vector<int> gray_val(bin_num);
    vector<float> target_feature;
    vector<char*> db_names;             // names of images in the database
    vector<vector<float>> db_fatures;   // features of all images in the database
    float distance;                     // distance between selected image from db and target
    map<char*, float> distance_map;     // dictionary that stores distances between images and target
    vector<pair<char*, float>> sorted_distance_map;

    // extract features from target image
    
    extract_grayscale(target, bin_num, gray_val);

    for (int i = 0; i < bin_num; i++) {
        target_feature.push_back(gray_val[i]);
    }
    // store all features and read them out
    append_all(dir, csv_file, 7, bin_num);
    read_image_data_csv(csv_file, db_names, db_fatures, false);

    // calculate all distances and stores them into distance_map
    for (int img_idx = 0; img_idx < db_fatures.size(); img_idx++) {
        distance = 0;
        for (int feature_idx = 0; feature_idx < db_fatures[img_idx].size(); feature_idx++) {
            distance += target_feature[feature_idx] - min(db_fatures[img_idx][feature_idx], target_feature[feature_idx]);
        }
        distance_map.insert(pair<char*, float>(db_names[img_idx], distance));
    }

    // sort the map and output the top 3 matches
    sort(distance_map, sorted_distance_map);
    printf("Extension: according to grayscale histogram matching, the top three matches to %s are:\n", sorted_distance_map[0].first);
    for (int i = 1; i <= 3; i++) {
        printf(sorted_distance_map[i].first);
        string tmp(sorted_distance_map[i].first);
        top3.push_back(tmp);
        printf("\n");
        // cout << sorted_distance_map[i].first << " " << sorted_distance_map[i].second << endl;
    }
    printf("\n");
    return 0;
}


int LOG_matching(Mat target, char *dir, char *csv_file, int bin_num, vector<string> &top3) {

    Mat target_hist;
    vector<float> target_feature;
    vector<int> tmp;
    vector<char*> db_names;             // names of images in the database
    vector<vector<float>> db_fatures;   // features of all images in the database
    float distance;                     // distance between selected image from db and target
    map<char*, float> distance_map;     // dictionary that stores distances between images and target
    vector<pair<char*, float>> sorted_distance_map;
    
    Mat log_ker = (Mat_<char>(5, 5) << 1, 0, -2, 0, -1,
                                        0, 0, 0, 0, 0, 
                                        -2, 0, 4, 0, -2, 
                                        0, 0, 0, 0, 0, 
                                        1, 0, -2, 0, 1);

    Mat filtered_img;
    filter2D(target, filtered_img, target.depth(), log_ker);
    int dim3d[3] = {bin_num, bin_num, bin_num};

    // extract features from target image
    target_hist = Mat::zeros(3, dim3d, CV_32S);
    
    extract_histogram_rgb(filtered_img, target_hist, bin_num, tmp);
    
    for (int i = 0; i < bin_num; i++) {
        for (int j = 0; j < bin_num; j++) {
            for (int k = 0; k < bin_num; k++) {
                target_feature.push_back(target_hist.at<int>(i, j, k));
            }
        }
    }
    
    // store all features and read them out
    append_all(dir, csv_file, 8, bin_num);
    read_image_data_csv(csv_file, db_names, db_fatures, false);

    // calculate all distances and stores them into distance_map
    for (int img_idx = 0; img_idx < db_fatures.size(); img_idx++) {
        distance = 0;
        for (int feature_idx = 0; feature_idx < db_fatures[img_idx].size(); feature_idx++) {
            distance += target_feature[feature_idx] - min(db_fatures[img_idx][feature_idx], target_feature[feature_idx]);
        }
        distance_map.insert(pair<char*, float>(db_names[img_idx], distance));
    }

    // sort the map and output the top 3 matches
    sort(distance_map, sorted_distance_map);
    printf("Extension: according to LOG matching, the top three matches to %s are:\n", sorted_distance_map[0].first);
    for (int i = 1; i <= 3; i++) {
        printf(sorted_distance_map[i].first);
        string tmp(sorted_distance_map[i].first);
        top3.push_back(tmp);
        printf("\n");
    }
    printf("\n");
    return 0;
}