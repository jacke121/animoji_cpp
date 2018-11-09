#include <math.h>
#include <algorithm>
#include <assert.h>
#include <cstring>
#include <memory>
#include "math_functions.hpp"

namespace  galaxy {
    void conv_forward(const Blob* input, Blob*& output, const Blob* w,
                      const Blob* b, pthreadpool_t threadpool, int pad0,
                      int pad1, int stride, bool activation) {
        assert(input->num_axes() == 4);
        assert(w->num_axes() == 4);
        assert(b->num_axes() == 1);

        Shape input_shape = input->shape();
        Shape kernel_shape_ = w->shape();
        assert(w->shape(1) == input->shape(1));

        int	batch_size = input_shape[0];
        int	image_channel = input_shape[1];
        int	image_row = input_shape[2];
        int	image_col = input_shape[3];
        int height = (image_row - kernel_shape_[2] + pad0 + pad1)/stride + 1;
        int width = (image_col - kernel_shape_[3] + pad0 + pad1)/stride + 1;
        Shape out_shape = { batch_size, kernel_shape_[0], height, width };
        if (output) {
            assert(w->shape(0) == output->shape(1));
            output->reshape(out_shape);;
        }
        else {
            output = new Blob(out_shape);
        }

        float* p_top = output->data();
        float* p_bottom = input->data();
        float* p_w = w->data();
        float* p_b = b->data();

        struct nnp_size input_size = {size_t(image_col),size_t(image_row) };
        struct nnp_padding input_padding = { size_t(pad0),size_t(pad1),size_t(pad1),size_t(pad0)};
        struct nnp_size kernel_size = { size_t(kernel_shape_[3]), size_t(kernel_shape_[2])};
        enum nnp_activation activation_ = activation ?
                                          nnp_activation_relu:
                                          nnp_activation_identity;

        if (batch_size == 1){
            struct nnp_size stride_ = {size_t(stride), size_t(stride)};
            nnp_convolution_inference(nnp_convolution_algorithm_auto,
                                      nnp_convolution_transform_strategy_tuple_based,
                                      size_t(image_channel), size_t(kernel_shape_[0]), input_size,
                                      input_padding, kernel_size, stride_, p_bottom,
                                      p_w, p_b, p_top, NULL, NULL, activation_,
                                      NULL, threadpool, NULL);
        }
        else{
            if (stride == 1 && batch_size > 3){
                nnp_convolution_output(nnp_convolution_algorithm_auto, size_t(batch_size),
                                       size_t(image_channel), size_t(kernel_shape_[0]), input_size,
                                       input_padding, kernel_size, p_bottom,
                                       p_w, p_b, p_top, NULL, NULL, activation_,
                                       NULL, threadpool, NULL);
            }
            else{
                int nb = input->count()/batch_size;
                int nt = output->count()/batch_size;
                struct nnp_size stride_ = {size_t(stride), size_t(stride)};
                for(int i = -batch_size; i; ++i){
                    nnp_convolution_inference(nnp_convolution_algorithm_auto,
                                              nnp_convolution_transform_strategy_tuple_based,
                                              size_t(image_channel), size_t(kernel_shape_[0]), input_size,
                                              input_padding, kernel_size, stride_, p_bottom,
                                              p_w, p_b, p_top, NULL, NULL, activation_,
                                              NULL, threadpool, NULL);
                    p_bottom += nb;
                    p_top += nt;
                }
            }
        }
    }

    void cnn_maxpooling(const Blob* input, Blob*& output, int size, int stride,
                        pthreadpool_t threadpool, padType pad_type) {
        assert(input->num_axes() == 4);

        Shape input_shape = input->shape();
        int	batch_size = input_shape[0];
        int	k = input_shape[1];
        int	row = input_shape[2];
        int	col = input_shape[3];
        int pad0, pad1, pad2, pad3;
        switch(pad_type){
            case None:{ // 0;
                pad0 = 0;
                pad1 = 0;
                pad2 = 0;
                pad3 = 0;
                break;
            }
            case Same:{ //same
                int pad_height = ((row - 1) / stride)*stride + size - row;

                int	pad_weight = ((col - 1) / stride)*stride + size - col;
                pad_height = (std::max)(0, pad_height);
                pad_weight = (std::max)(0, pad_weight);
                pad0 = pad_height / 2;
                pad1 = pad_height - pad0;
                pad2 = pad_weight / 2;
                pad3 = pad_weight - pad2;
                break;
            }
            case Valid:{ //valid
                int pad_height = ((row - size + 1) / stride)*stride + size - row;
                int	pad_weight = ((col - size + 1) / stride)*stride + size - col;
                pad_height = (std::max)(0, pad_height);
                pad_weight = (std::max)(0, pad_weight);
                pad0 = pad_height / 2;
                pad1 = pad_height - pad0;
                pad2 = pad_weight / 2;
                pad3 = pad_weight - pad2;
                break;
            }
            default:{
                printf("Unknow padding type.");
                exit(1);
            }
        }
        int height = (row + pad0 + pad1 - size) / stride + 1;
        int width = (col + pad2 + pad3 - size) / stride + 1;
        Shape out_shape = { batch_size, k, height, width };
        if (output) {
            output->reshape(out_shape);
        }
        else {
            output = new Blob(out_shape);
        }
        float* p_top = output->data();
        float* p_bottom = input->data();
        struct nnp_size input_size = {size_t(col), size_t(row)};
        struct nnp_padding input_padding = { size_t(pad0), size_t(pad3), size_t(pad1), size_t(pad2)};
        struct nnp_size pool_size = { size_t(size),size_t(size)};
        struct nnp_size pool_stride = {size_t(stride), size_t(stride)};
        nnp_max_pooling_output(size_t(batch_size), size_t(k), input_size, input_padding, pool_size,
                               pool_stride, p_bottom, p_top, threadpool);
    }

    void fully_connected(const Blob* input, Blob*& output, const Blob* w, const Blob* b,
                         pthreadpool_t threadpool) {
        assert(input->num_axes() == 2 || input->num_axes() == 4);
        assert(input->count()/input->shape(0) == w->shape(1));

        Shape input_shape = input->shape();
        if (input_shape.size() == 4) {
            input_shape = { input_shape[0], input_shape[1] * input_shape[2] * input_shape[3] };
        }

        int	batch_size = input_shape[0];
        int filters = w->shape(0);
        int input_dim = w->shape(1);

        Shape out_shape = { batch_size,filters };
        if (output) {
            output->reshape(out_shape);
        }
        else {
            output = new Blob(out_shape);
        }

        float* p_top = output->data();
        float* p_bottom = input->data();
        float* p_w = w->data();
        float* p_b = b->data();

        if (batch_size == 1){
            nnp_fully_connected_inference(size_t(input_dim), size_t(filters),
                                          p_bottom, p_w, p_top, threadpool);
        }
        else{
            nnp_fully_connected_output(size_t(batch_size), size_t(input_dim), size_t(filters),
                                       p_bottom, p_w, p_top, threadpool, NULL);
        }
        for(int i = -batch_size; i; ++i){
            float* p_b_i = p_b;
            for(int j = -filters; j; ++j){
                *p_top++ += *p_b_i++;
            }
        }
    }

    void softmax(Blob* input, pthreadpool_t threadpool) {
        Shape shape = input->shape();
        if(shape.size() == 4){
            int	row = shape[2];
            int	col = shape[3];
            assert(shape[1] == 2);
            int loopCount = row*col;
            float* data = input->data();
            for(int b = -shape[0]; b; ++b){
                float* a1 = data;
                float* a2 = a1 + loopCount;
                data = a2 + loopCount;
                for (int i = -loopCount; i; ++i) {
                    register float a1_ = *a1;
                    register float a2_ = *a2;
                    register float max = a1_ > a2_ ? a1_ : a2_;
                    a1_ = exp(a1_ - max);
                    a2_ = exp(a2_ - max);
                    register float sum = 1.0f/(a1_ + a2_);
                    *a1++ = a1_ * sum;
                    *a2++ = a2_ * sum;
                }
            }
        }
        else{
            assert(shape[1] == 2);
            float* data = input->data();
            nnp_softmax_output(size_t(shape[0]), size_t(shape[1]), data, data, threadpool);
        }
    }

    void leaky(Blob* input, pthreadpool_t threadpool, float alpha) {
        float* data = input->data();
        int batch_size = input->shape(0);
        int c = input->count()/batch_size;
        nnp_relu_output(size_t(batch_size), size_t(c), data, data, alpha, threadpool);
    }

    void prelu(Blob* input, const Blob* alphas) {
        Shape shape = input->shape();
        assert(shape.size() == 2 || shape.size() == 4);
        int	batch_size = shape[0];
        int	k = shape[1];
        int hw = shape.size() == 2 ? 1 : shape[2] * shape[3];

        float* data = input->data();
        for (int bs = -batch_size; bs; ++bs) {
            float* alphas_data = alphas->data();
            for (int c = -k; c; ++c) {
                register float alphas_value = *alphas_data++;
                for (int i = -hw; i; ++i) {
                    if (*data < 0) (*data) *= alphas_value;
                    data++;
                }
            }
        }
    }

    void nms(std::vector<bbox>& boxes, float thresh, bool IsMin) {
        sort(boxes.begin(), boxes.end(), [](bbox a, bbox b) {return a.score > b.score; });
        int num_box = static_cast<int>(boxes.size());
        if (num_box == 0) {
            boxes.resize(0);
            return;
        }
        float* area = new float[num_box];
        for (int i = 0; i < num_box; ++i) {
            area[i] = static_cast<float>((boxes[i].x2 - boxes[i].x1 + 1)*(boxes[i].y2 - boxes[i].y1 + 1));
        }
        int idx = 0;
        for (int i = 0; i < num_box; ++i) {
            if (boxes[i].score > 0) {
                for (int j = i + 1; j < num_box; ++j) {
                    if (boxes[j].score > 0) {
                        int xx1 = (std::max)(boxes[i].x1, boxes[j].x1);
                        int yy1 = (std::max)(boxes[i].y1, boxes[j].y1);
                        int xx2 = (std::min)(boxes[i].x2, boxes[j].x2);
                        int yy2 = (std::min)(boxes[i].y2, boxes[j].y2);
                        int w = (std::max)(0, xx2 - xx1 + 1);
                        int h = (std::max)(0, yy2 - yy1 + 1);
                        float inter = static_cast<float>(w*h);
                        float U = IsMin ? (std::min)(area[i], area[j])
                                        :(area[i] + area[j] - inter);
                        float ovr = inter / U;
                        if (ovr > thresh) boxes[j].score = -1;
                    }
                }
                if(idx != i) boxes[idx] = boxes[i];
                idx++;
            }
        }
        delete[] area;
        boxes.resize(size_t(idx));
    }

}//namespace  galaxy
