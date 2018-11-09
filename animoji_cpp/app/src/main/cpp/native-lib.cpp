#include <jni.h>
#include "face_prediction.h"
#include "detection.hpp"

using namespace galaxy;
extern "C" JNIEXPORT jfloat
JNICALL
Java_com_tcl_animoji_1cpp_MainActivity_FacePredictionFromJNI(
        JNIEnv *env,
        jobject/* this */
       ) {
    //std::string hello = "Hello from C++";
    //return env->NewStringUTF(hello.c_str());
    return face_prediction();
}
extern "C" JNIEXPORT jfloat
JNICALL
Java_com_tcl_animoji_1cpp_MainActivity_getDetectTimeFromJNI(
        JNIEnv *env,
jobject/* this */) {
//     DetectNet  Detect_1;
    extern float detect_time;
    return detect_time;
}
extern "C" JNIEXPORT jfloat
JNICALL
Java_com_tcl_animoji_1cpp_MainActivity_getLandmarkTimeFromJNI(
        JNIEnv *env,
        jobject/* this */) {
//      DetectNet Detect_2;

    extern float landmark_time;
    return landmark_time;
}