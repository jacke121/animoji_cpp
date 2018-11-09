#include <assert.h>
#include <algorithm>
#include <memory>
#include "landmark.hpp"
#include "math_functions.hpp"

namespace galaxy {
    inline void _convert_to_square(std::vector<bbox>& boxes, float expand=0.0f){
        for (size_t i = 0; i < boxes.size(); ++i) {
            int w = boxes[i].x2 - boxes[i].x1 + 1;
            int h = boxes[i].y2 - boxes[i].y1 + 1;
            int max_side = static_cast<int>(((std::max)(h, w))*(1 + expand) + 0.5);
            boxes[i].x1 = static_cast<int>(boxes[i].x1 + w*0.5 - max_side*0.5 + 0.5);
            boxes[i].y1 = static_cast<int>(boxes[i].y1 + h*0.5 - max_side*0.5 + 0.5);
            boxes[i].x2 = boxes[i].x1 + max_side - 1;
            boxes[i].y2 = boxes[i].y1 + max_side - 1;
        }
    }

    inline void _pad(const std::vector<bbox>& boxes, int* return_list,
                     int width, int height) {
        for (size_t i = 0; i < boxes.size(); ++i) {
            int& dy = *return_list++;
            int& edy = *return_list++;
            int& dx = *return_list++;
            int& edx = *return_list++;
            int& y = *return_list++;
            int& ey = *return_list++;
            int& x = *return_list++;
            int& ex = *return_list++;

            x = boxes[i].x1;
            y = boxes[i].y1;
            ex = boxes[i].x2 + 1;
            ey = boxes[i].y2 + 1;

            if (ex > width) { edx = ex - width; ex = width; } else edx = 0;
            if (ey > height) { edy = ey - height; ey = height; } else edy = 0;
            if (x < 0) { dx = -x; x = 0; } else dx = 0;
            if (y < 0) { dy = -y; y = 0; } else dy = 0;
        }
    }

    LandmarkNet::LandmarkNet(pthreadpool_t threadpool)
        :threadpool_(threadpool){
        build_net();
    }


    void LandmarkNet::load_weight(std::ifstream& infile) {
        for (size_t i = 0; i < param_.size(); ++i) {
            infile.read((char*)param_[i]->data(), param_[i]->count()*sizeof(float));
            assert(infile.gcount() == param_[i]->count()*sizeof(float));
        }
    }

    void LandmarkNet::build_net(){
        param_.reserve(23);
        param_.push_back(new Blob(32, 3, 3, 3));
        param_.push_back(new Blob(32));
        param_.push_back(new Blob(32));

        param_.push_back(new Blob(64, 32, 3, 3));
        param_.push_back(new Blob(64));
        param_.push_back(new Blob(64));

        param_.push_back(new Blob(64, 64, 3, 3));
        param_.push_back(new Blob(64));
        param_.push_back(new Blob(64));

        param_.push_back(new Blob(128, 64, 2, 2));
        param_.push_back(new Blob(128));
        param_.push_back(new Blob(128));

        param_.push_back(new Blob(256, 128*3*3));
        param_.push_back(new Blob(256));
        param_.push_back(new Blob(256));

        param_.push_back(new Blob(2, 256));
        param_.push_back(new Blob(2));

        param_.push_back(new Blob(4, 256));
        param_.push_back(new Blob(4));

        param_.push_back(new Blob(10, 256));
        param_.push_back(new Blob(10));

        param_.push_back(new Blob(140, 256));
        param_.push_back(new Blob(140));

        blobs_.resize(13);
    #ifdef _DEBUG
        for (int i = 0; i < 13; ++i)
            assert(!blobs_[i]);
    #endif
    }

    void LandmarkNet::forward(const Blob* input) {
        /*

          void conv_forward(const Blob* input, Blob*& output, const Blob* w,
                      const Blob* b, pthreadpool_t threadpool, int pad0,
                      int pad1, int stride, bool activation){}

        */
        conv_forward(input, blobs_[0], param_[0], param_[1], threadpool_);

        cnn_maxpooling(blobs_[0], blobs_[1], 3, 2, threadpool_, Same);
        prelu(blobs_[1], param_[2]);

        conv_forward(blobs_[1], blobs_[2], param_[3], param_[4], threadpool_);
        cnn_maxpooling(blobs_[2], blobs_[3], 3, 2, threadpool_, Valid);
        prelu(blobs_[3], param_[5]);

        conv_forward(blobs_[3], blobs_[4], param_[6], param_[7], threadpool_);
        cnn_maxpooling(blobs_[4], blobs_[5], 2, 2, threadpool_, Same);
        prelu(blobs_[5], param_[8]);

        conv_forward(blobs_[5], blobs_[6], param_[9], param_[10], threadpool_);
        prelu(blobs_[6], param_[11]);

        fully_connected(blobs_[6], blobs_[7], param_[12], param_[13], threadpool_);
        prelu(blobs_[7], param_[14]);

        fully_connected(blobs_[7], blobs_[8], param_[15], param_[16], threadpool_);
        softmax(blobs_[8], threadpool_);

        fully_connected(blobs_[7], blobs_[9], param_[17], param_[18], threadpool_);

        fully_connected(blobs_[7], blobs_[10], param_[19], param_[20], threadpool_);

        fully_connected(blobs_[7], blobs_[11], param_[21], param_[22], threadpool_);
    }

    void LandmarkNet::predict(const cv::Mat& im, std::vector<bbox>& boxes) {
        const float threshold = 0.7;
        const int net_size = 48;
        const int& height = im.rows;
        const int& width = im.cols;
        _convert_to_square(boxes, 0.3);
        int nbox = boxes.size();
        int* return_list = new int[nbox * 8];
        _pad(boxes, return_list, width, height);
        Blob* input = new Blob(nbox, 3, net_size, net_size);
        int hw = net_size*net_size;
        float* input_data = input->data();

        int* return_list_tmp = return_list;
        for (int i = -nbox; i; ++i) {
            int dy = *return_list_tmp++;
            int edy = *return_list_tmp++;
            int dx = *return_list_tmp++;
            int edx = *return_list_tmp++;
            int y = *return_list_tmp++;
            int ey = *return_list_tmp++;
            int x = *return_list_tmp++;
            int ex = *return_list_tmp++;

            cv::Mat roi_img = im(cv::Range(y, ey), cv::Range(x, ex));
            if(dy > 0 || edy > 0 || dx > 0 || edx > 0)
                cv::copyMakeBorder(roi_img, roi_img, dy, edy, dx, edx,
                               cv::BORDER_CONSTANT, 0);

            cv::resize(roi_img, roi_img, cv::Size(net_size, net_size), CV_INTER_LINEAR);
            std::vector<cv::Mat> bgr;
            cv::split(roi_img, bgr);
            cv::Mat tmp_mat = cv::Mat(cv::Size(net_size, net_size), CV_32FC1, input_data);
            for (size_t bgr_ = 0; bgr_ < bgr.size(); ++bgr_) {
                bgr[bgr_].convertTo(tmp_mat, CV_32FC1, 1.0f/128, -127.5f/128);
                input_data += hw;
                tmp_mat.data = static_cast<uchar *>((void*)input_data);
            }
        }

        forward(input);
        float* cls_scores = blobs_[8]->data();
        float* reg = blobs_[9]->data();
        float* landmark = blobs_[10]->data();
        float* animoji = blobs_[11]->data();
        int out_idx = 0;
        for (int k = 0; k < nbox; ++k) {
            bbox& box = boxes[k];
            float scores = cls_scores[2 * k + 1];
            if (scores > threshold) {
                int w = box.x2 - box.x1 + 1;
                int h = box.y2 - box.y1 + 1;
                float* offset = reg + 4 * k;

                int x1 = (std::max)(0, box.x1+static_cast<int>(*offset++*w+0.5));
                int y1 = (std::max)(0, box.y1+static_cast<int>(*offset++*h+0.5));
                int x2 = (std::min)(width, box.x2+static_cast<int>(*offset++*w+0.5));
                int y2 = (std::min)(height, box.y2+static_cast<int>(*offset++*h+0.5));
                if (x2 > x1 && y2 > y1){
                    bbox& out_box = boxes[out_idx++];
                    float* landmark_i = landmark + 10 * k;
                    float* animoji_i = animoji + 140 * k;
                    float* out_box_landmark = out_box.create_array(150);
                    for (int l = 5; l; --l) {
                        *out_box_landmark++ = w* *landmark_i++ + box.x1 - 1;
                        *out_box_landmark++ = h* *landmark_i++ + box.y1 - 1;
                    }
                    for (int l = 70; l; --l) {
                        *out_box_landmark++ = w* *animoji_i++ + box.x1 - 1;
                        *out_box_landmark++ = h* *animoji_i++ + box.y1 - 1;
                    }

                    out_box.x1 = x1;
                    out_box.y1 = y1;
                    out_box.x2 = x2;
                    out_box.y2 = y2;
                    out_box.score = scores;
                }
            }
        }
        delete[] return_list;
        delete input;
        boxes.resize(out_idx);
        if(out_idx > 1) nms(boxes, 0.6, true);
    }

    LandmarkNet::~LandmarkNet(){
        for (size_t i = 0; i < blobs_.size(); ++i) {
            delete blobs_[i];
        }
        for (size_t i = 0; i < param_.size(); ++i) {
            delete param_[i];
        }
    }


} // galaxy
