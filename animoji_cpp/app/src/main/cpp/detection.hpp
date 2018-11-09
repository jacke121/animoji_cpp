#ifndef DETECTION_HPP_
#define DETECTION_HPP_

#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "blob.hpp"
#include <nnpack.h>
#include <pthreadpool.h>
#include "landmark.hpp"

namespace  galaxy {
    class DetectNet {
    public:
        DetectNet(int num_threads = -1);
        void build_net();
        void forward(const Blob* input);
        void load_weight(const std::string& model_path);
        std::vector<bbox> predict(const cv::Mat& im);
        ~DetectNet();
//        float getDetectTime();
//        float getLandmarkTime();
//        float detect_time,landmark_time;

    protected:
        pthreadpool_t threadpool_;
        LandmarkNet*  landmarknet_;
        std::vector<Blob*> param_;
        std::vector<Blob*> blobs_;
    };

} //namespace  galaxy
#endif //DETECTION_HPP_
