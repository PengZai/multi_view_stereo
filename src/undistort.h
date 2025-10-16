#pragma once
#include "configs.h"
#include "datasets/dataset.h"
#include "utils.h"

namespace MVS{

class Image;  // forward declaration

class Undistort
{
    public:

    Undistort();
    ~Undistort();


    void createRemapWithPinholeAndRadtan(int width, int height, const std::vector<float>& intrinsics, const std::vector<float>& distortion_coeffs);
    void createRemapWithPinholeAndFOV(int width, int height, const std::vector<float>& intrinsics, const std::vector<float>& distortion_coeffs);
    void createRemapWithUndistortedModel(int width, int height);
    void run(const Image* image);


    protected:
    float* remapU_;
    float* remapV_;


};


}


