#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <chrono>
#include <vector>
#include <unistd.h>
#include <dirent.h>
#include <iostream>
#include "detection.hpp"

using namespace galaxy;
using namespace std::chrono;

float face_prediction() {

    high_resolution_clock::time_point BeginTime, EndTime;
    BeginTime = high_resolution_clock::now();
    DetectNet detect(-1);
    detect.load_weight("/storage/emulated/0/DCIM/Camera/detect_landmark.bin");
    EndTime = high_resolution_clock::now();
    float build_time = (float)duration_cast<microseconds>(EndTime - BeginTime).count()*1e-3;
    std::cout << "Build model use time: " << build_time << " ms" << std::endl;
    cv::Mat im;

    const int avg_n = 50;
    float* avg_delay = new float[avg_n];
    for(int i = 0; i < avg_n; ++i) avg_delay[i] = -1;
    int offset = 0;

//    while(true){
    float dt=0;
    im = cv::imread("/storage/emulated/0/DCIM/Camera/1.jpg");
//    for(int num=0;num<avg_n;++num){
//        BeginTime = high_resolution_clock::now();
//        std::vector<bbox> boxes = detect.predict(im);
//
//        EndTime = high_resolution_clock::now();
//        dt = (float)duration_cast<microseconds>(EndTime - BeginTime).count()*1e-3;
//
//        avg_delay[offset] = dt;
//        offset = (offset + 1) % avg_n;
//        }
//    float avg_dt = 0.0f;
//    int count = 0;
//    for(int i =0; i< avg_n; ++i){
//        if(avg_delay[i] > 0){
//            avg_dt += avg_delay[i];
//            count++;
//        }
//    }
//    avg_dt /= count;
//    std::cout << dt  << ", " << avg_dt << std::endl;

    std::vector<bbox> boxes = detect.predict(im);
    for (size_t i = 0; i < boxes.size(); ++i) {
        cv::Scalar color=cv::Scalar(0,255,0);
        float* landmark = boxes[i].array() + 10;
        cv::rectangle(im, cv::Rect(boxes[i].x1, boxes[i].y1,
                boxes[i].x2 - boxes[i].x1, boxes[i].y2 - boxes[i].y1),
                      color, 2, 1, 0);

        if (landmark)
        for (int j = 0; j < 70; ++j){
            cv::circle(im, cv::Point((int)(0.5+landmark[2*j]), (int)(0.5+landmark[2*j+1])),
                    int(0.1), color, 2, 1, 0);
        }
    }
//    cv::imshow("0", im);
    cv::imwrite("/storage/emulated/0/DCIM/Camera/result_1.jpg", im);
//        int c = cv::waitKey(10);
//        if(c == 27) break;

//    }
    delete[]avg_delay;
    //return avg_dt;
    return build_time;
}

//float getDetectTime(){
//    extern float detect_time;
//  return detect_time;
//}
//float getLandmarkTime(){
//    extern float landmark_time;
//    return landmark_time;
//}