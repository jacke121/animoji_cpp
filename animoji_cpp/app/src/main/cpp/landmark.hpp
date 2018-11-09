#ifndef LANDMARK_HPP_
#define LANDMARK_HPP_

#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "blob.hpp"

#include <nnpack.h>
#include <pthreadpool.h>

namespace  galaxy {
    class LandmarkNet {
    public:
        LandmarkNet(pthreadpool_t threadpool);
        void build_net();
        void forward(const Blob* input);
        void load_weight(std::ifstream& infile);
        void predict(const cv::Mat& im, std::vector<bbox>& boxes);
        ~LandmarkNet();

    protected:
        pthreadpool_t threadpool_;
        std::vector<Blob*> param_;
        std::vector<Blob*> blobs_;
    };
} //namespace  galaxy
#endif //LANDMARK_HPP_

//float getLandmarkTime();
//float getDetectTime();