# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++0x")

include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/nnpack/include)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/pthreadpool/include)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/opencv/native/jni/include)

add_library( # Sets the name of the library.
             native-lib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             src/main/cpp/native-lib.cpp
             src/main/cpp/blob.cpp
             src/main/cpp/detection.cpp
             src/main/cpp/landmark.cpp
             src/main/cpp/math_functions.cpp
             src/main/cpp/face_prediction.cpp)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

 add_library(nnpack STATIC IMPORTED)
  set_target_properties(nnpack
    PROPERTIES IMPORTED_LOCATION
    E:/TCL_code/android/animoji_cpp/app/libs/armeabi-v7a/libnnpack.so)

 add_library(opencv_java3 STATIC IMPORTED)
  set_target_properties(opencv_java3
     PROPERTIES IMPORTED_LOCATION
     E:/TCL_code/android/animoji_cpp/app/libs/armeabi-v7a/libopencv_java3.so)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                       native-lib
                       nnpack
                       opencv_java3

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib})