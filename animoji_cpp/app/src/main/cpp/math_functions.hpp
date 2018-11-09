#ifndef MATH_FUNCTIONS_HPP_
#define MATH_FUNCTIONS_HPP_
#include "blob.hpp"
#include <nnpack.h>
#include <pthreadpool.h>

namespace  galaxy {
    enum padType {None, Valid, Same};
    void conv_forward(const Blob* input, Blob*& output, const Blob* w,
                      const Blob* b, pthreadpool_t threadpool, int pad0=0,
                      int pad1=0, int stride=1, bool activation=false);

    void cnn_maxpooling(const Blob* input, Blob*& output, int size, int stride,
                        pthreadpool_t threadpool, padType pad_type = Same);

    void fully_connected(const Blob* input, Blob*& output, const Blob* w, const Blob* b,
                             pthreadpool_t threadpool);
    void softmax(Blob* input, pthreadpool_t threadpool);
    void leaky(Blob* input, pthreadpool_t threadpool, float alpha = 0.1f);
    void prelu(Blob* input, const Blob* alphas);
    void nms(std::vector<bbox>& boxes, float thresh, bool IsMin=false);
} //namespace  galaxy
#endif //MATH_FUNCTIONS_HPP_
