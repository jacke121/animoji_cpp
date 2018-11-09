#include <assert.h>
#include <algorithm>
#include <memory>
#include <thread>
#include "math_functions.hpp"
#include "detection.hpp"
#include "landmark.hpp"

using namespace std::chrono;
float detect_time,landmark_time;
namespace galaxy {

    static float anchors[10] = {
        0.72213f,      1.12904f,
        1.27083f,      2.02340f,
        2.213495f,     3.44419f,
        3.60766f,      5.54824f,
        5.538638f,     8.54274f
    };

    inline void sigmoid(float* a, int count){
        for(int i = -count; i; ++i){
            *a = *a > 20.0f ? 1.0f:(*a < -20.0f ? 0.0f:1.0f/(1.0f+exp(-*a)));
            a++;
        }
    }

    std::vector<bbox> generate_bbox(const Blob* feature_map,
                    const int im_height, const int im_width) {

        int nbox = 5;
        float thresh = 0.40f;
        Shape shape = feature_map->shape();
        int batch_size = shape[0];
        assert(shape[1] == 6*nbox);
        int height = shape[2];
        int width = shape[3];
        int step = height*width;
        float scale_width = im_width/width;
        float scale_height = im_height/height;

        float* data = feature_map->data();
        std::vector<bbox> boxes;
        boxes.reserve(step*nbox);
        for (int bs = -batch_size; bs; ++bs){
            for(int b = 0; b < nbox; ++b){
                float* x = data;
                float* y = x + step;
                float* w = y + step;
                float* h = w + step;
                float* conf = h + step;
                data = conf + 2*step;
                sigmoid(conf, step);
                int n = 0;
                for (int i = 0; i < height; ++i){
                    for(int j = 0; j < width; ++j){
                        if(conf[n] > thresh){
                            float xx = x[n] > 20.0f ? 1.0f:(x[n] < -20.0f? 0.0f:1.0f/(1.0f+exp(-x[n])));
                            float yy = y[n] > 20.0f ? 1.0f:(y[n] < -20.0f? 0.0f:1.0f/(1.0f+exp(-y[n])));

                            xx = scale_width*(xx + j);
                            yy = scale_height*(yy + i);

                            float ww = scale_width*(exp(w[n])*anchors[2 * b])*0.5f;
                            float hh = scale_height*(exp(h[n])*anchors[2 * b + 1])*0.5f;

                            register int x1 = (std::max)(0, static_cast<int>(xx - ww + 0.5f));
                            register int y1 = (std::max)(0, static_cast<int>(yy - hh + 0.5f));
                            register int x2 = (std::min)(im_width, static_cast<int>(xx + ww + 0.5f));
                            register int y2 = (std::min)(im_height, static_cast<int>(yy + hh + 0.5f));
                            if(x2 > x1 && y2 > y1)
                                boxes.emplace_back(bbox(x1,y1,x2,y2,conf[n]));
                        }
                        n++;
                    }
                }
            }
        }

        boxes.resize(boxes.size());
        nms(boxes, 0.6, false);
        return boxes;
    }

    void DetectNet::build_net(){
        param_.reserve(28);
        param_.push_back(new Blob(8, 3, 3, 3));
        param_.push_back(new Blob(8));

        param_.push_back(new Blob(12, 8, 3, 3));
        param_.push_back(new Blob(12));

        param_.push_back(new Blob(16, 12, 3, 3));
        param_.push_back(new Blob(16));

        param_.push_back(new Blob(8, 16, 1, 1));
        param_.push_back(new Blob(8));

        param_.push_back(new Blob(16, 8, 3, 3));
        param_.push_back(new Blob(16));

        param_.push_back(new Blob(32, 16, 3, 3));
        param_.push_back(new Blob(32));

        param_.push_back(new Blob(16, 32, 1, 1));
        param_.push_back(new Blob(16));

        param_.push_back(new Blob(32, 16, 3, 3));
        param_.push_back(new Blob(32));

        param_.push_back(new Blob(64, 32, 3, 3));
        param_.push_back(new Blob(64));

        param_.push_back(new Blob(32, 64, 1, 1));
        param_.push_back(new Blob(32));

        param_.push_back(new Blob(64, 32, 3, 3));
        param_.push_back(new Blob(64));

        param_.push_back(new Blob(32, 64, 1, 1));
        param_.push_back(new Blob(32));

        param_.push_back(new Blob(64, 32, 3, 3));
        param_.push_back(new Blob(64));

        param_.push_back(new Blob(30, 64, 1, 1));
        param_.push_back(new Blob(30));

        blobs_.resize(18);
#ifdef _DEBUG
        for (int i = 0; i < 18; ++i)
            assert(!blobs_[i]);
#endif
    }

    DetectNet::DetectNet(int num_threads){
        enum nnp_status init_status = nnp_initialize();
        if (init_status != nnp_status_success) {
                fprintf(stderr, "Initialization failed: error code %d\n", init_status);
            exit(EXIT_FAILURE);
        }

        int nMaxThreads = std::thread::hardware_concurrency();
        int nThreads;
        if((num_threads <=0 || num_threads > nMaxThreads)){
            nThreads = nMaxThreads;
        }
        else{
            nThreads = num_threads;
        }
//        printf("nThreads = %d \n", nThreads);
        threadpool_ = pthreadpool_create(nThreads);
        build_net();
        landmarknet_ = new LandmarkNet(threadpool_);
    }

    void DetectNet::load_weight(const std::string& model_path) {
        std::ifstream infile(model_path.c_str(), std::ifstream::binary);
        if (!infile.is_open()) {
            std::cout << "Open file fail: " << model_path << std::endl;
            exit(1);
        }
        for (size_t i = 0; i < param_.size(); ++i) {
            infile.read((char*)param_[i]->data(), param_[i]->count()*sizeof(float));
            assert(infile.gcount() == param_[i]->count()*sizeof(float));
        }
        landmarknet_->load_weight(infile);
        infile.close();
    }

    void DetectNet::forward(const Blob* input){
        conv_forward(input, blobs_[0], param_[0], param_[1], threadpool_,
                1, 1, 1, false);

        cnn_maxpooling(blobs_[0], blobs_[1], 2, 2, threadpool_, None);
        leaky(blobs_[1], threadpool_);

        conv_forward(blobs_[1], blobs_[2], param_[2], param_[3], threadpool_,
                1, 1, 1, false);

        cnn_maxpooling(blobs_[2], blobs_[3], 2, 2, threadpool_, None);
        leaky(blobs_[3], threadpool_);

        conv_forward(blobs_[3], blobs_[4], param_[4], param_[5], threadpool_,
                1, 1, 1, false);
        leaky(blobs_[4], threadpool_);

        conv_forward(blobs_[4], blobs_[5], param_[6], param_[7], threadpool_,
                0, 0, 1, false);
        leaky(blobs_[5], threadpool_);

        conv_forward(blobs_[5], blobs_[6], param_[8], param_[9], threadpool_,
                1, 1, 1, false);

        cnn_maxpooling(blobs_[6], blobs_[7], 2, 2, threadpool_, None);
        leaky(blobs_[7], threadpool_);

        conv_forward(blobs_[7], blobs_[8], param_[10], param_[11], threadpool_,
                1, 1, 1, false);
        leaky(blobs_[8], threadpool_);

        conv_forward(blobs_[8], blobs_[9], param_[12], param_[13], threadpool_,
                0, 0, 1, false);
        leaky(blobs_[9], threadpool_);

        conv_forward(blobs_[9], blobs_[10], param_[14], param_[15], threadpool_,
                1, 1, 1, false);

        cnn_maxpooling(blobs_[10], blobs_[11], 2, 2, threadpool_, None);
        leaky(blobs_[11], threadpool_);

        conv_forward(blobs_[11], blobs_[12], param_[16], param_[17], threadpool_,
                1, 1, 1, false);
        leaky(blobs_[12], threadpool_);

        conv_forward(blobs_[12], blobs_[13], param_[18], param_[19], threadpool_,
                0, 0, 1, false);
        leaky(blobs_[13], threadpool_);

        conv_forward(blobs_[13], blobs_[14], param_[20], param_[21], threadpool_,
                1, 1, 1, false);
        leaky(blobs_[14], threadpool_);

        conv_forward(blobs_[14], blobs_[15], param_[22], param_[23], threadpool_,
                0, 0, 1, false);
        leaky(blobs_[15], threadpool_);

        conv_forward(blobs_[15], blobs_[16], param_[24], param_[25], threadpool_,
                1, 1, 1, false);
        leaky(blobs_[16], threadpool_);

        conv_forward(blobs_[16], blobs_[17], param_[26], param_[27], threadpool_,
                0, 0, 1, false);
    }

        std::vector<bbox> DetectNet::predict(const cv::Mat& im){
//            std::cout << im.rows << " " << im.cols << std::endl;
        //detect begin
            high_resolution_clock::time_point Detect_BeginTime,Detect_EndTime,Landmark_BeginTime,Landmark_EndTime;
            Detect_BeginTime= high_resolution_clock::now();
            const int input_dim = 112;
            cv::Mat dst;
            cv::resize(im, dst, cv::Size(input_dim, input_dim), CV_INTER_LINEAR);
            std::vector<cv::Mat> bgr;
            cv::split(dst, bgr);
            Blob* input = new Blob(1, 3, input_dim, input_dim);
            float* data = input->data();
            cv::Mat tmp_mat(cv::Size(input_dim, input_dim), CV_32FC1, data);
            for(int i = 3; i; --i){
                bgr[i-1].convertTo(tmp_mat, CV_32FC1, 1.0f/255);
                data += input_dim*input_dim;
                tmp_mat.data = static_cast<uchar *>((void*)data);
            }

            forward(input);
            delete input;
            std::vector<bbox> boxes = generate_bbox(blobs_[17], im.rows, im.cols);
        //detect end
            Detect_EndTime=high_resolution_clock::now();
            detect_time = (float)duration_cast<microseconds>(Detect_EndTime - Detect_BeginTime).count()*1e-3;
        //landmark begin
            Landmark_BeginTime=high_resolution_clock::now();
            if (boxes.size() > 0) landmarknet_->predict(im, boxes);
            Landmark_EndTime=high_resolution_clock::now();
        //landmark end
            landmark_time = (float)duration_cast<microseconds>(Landmark_EndTime - Landmark_BeginTime).count()*1e-3;
            return boxes;
        }

        DetectNet::~DetectNet(){
            delete landmarknet_;
            for (size_t i = 0; i < blobs_.size(); ++i) {
                delete blobs_[i];
            }
            for (size_t i = 0; i < param_.size(); ++i) {
                delete param_[i];
            }
        if(0 != pthreadpool_get_threads_count(threadpool_))
            pthreadpool_destroy(threadpool_);
        }
} // galaxy
