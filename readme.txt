CS5330 Project 2
Xichen Liu


Khoury wiki:
https://wiki.khoury.northeastern.edu/display/~xicliu/CS5330-Project-2-Report

cbir.cpp: main program
data.csv: stores the feature info
matching.cpp/matching.h: includes all functions used in cbir.cpp


Operating system: Windows 11
IDE: vscode
code-runner execute command: "cd $dir && g++ $fileName matching.cpp -o $fileNameWithoutExt -std=c++14 -I D:\\CodeAndTools\\OpenCV\\opencv\\build\\include -I D:\\CodeAndTools\\OpenCV\\opencv\\build\\include\\opencv2 -L D:\\CodeAndTools\\OpenCV\\opencv\\build\\x64\\MinGW\\lib -l opencv_core455 -l opencv_core455 -l opencv_imgproc455 -l opencv_imgcodecs455 -l opencv_video455 -l opencv_ml455 -l opencv_highgui455 -l opencv_objdetect455 -l opencv_flann455 -l opencv_imgcodecs455 -l opencv_photo455 -l opencv_videoio455 && $dir$fileNameWithoutExt"


Procedure of running cbir.cpp:

1. Enter the path of image that user wants to use as a target image
2. Enter the image base as the candidates of matching
3. Select the matching method that user wants to apply the target image. Options will be displayed when directory's name has entered.
4. Wait until results shown, task 4 may take a little longer.
5. The rank of samilarity of the result images are shown as the window titles.