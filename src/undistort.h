#pragma once
#include "configs.h"
#include "datasets/dataset.h"
#include "utils.h"

namespace MVS{

class Image;  // forward declaration

class DistortModel
{
    public:

    DistortModel(int width, int height, int wOrg, int hOrg);
    virtual ~DistortModel();

    void resetRemapUV();

    virtual void distortCoordinates(float* in_u, float* in_v, int n, float in_fx, float in_fy, float in_cx, float in_cy,
    float* out_u, float* out_v) = 0;

    void makeResizeK(std::vector<float> original_intrinsics, std::vector<float>& intrinsics);
    void makeOptimalKCrop(std::vector<float>& intrinsics);
  
    void undistort(Image* const image);


    protected:
    int wOrg_;
    int hOrg_;
    int width_;
    int height_;
    int wh_;

    float unit_fx_ = 1.0;
    float unit_fy_ = 1.0;
    float unit_cx_ = 0.0;
    float unit_cy_ = 0.0;

    float* remapU_;
    float* remapV_;


};


// class EquidistantDistortModel : public DistortModel
// {

//   public:

//   EquidistantDistortModel(int new_w, int new_h, int wOrg, int hOrg);

//   void distortCoordinates(float* in_u, float* in_v, int n, float in_fx, float in_fy, float in_cx, float in_cy,
//  float* out_u, float* out_v) override;

//   protected:
//   // 畸变参数
//   float k1 = -0.04345139283609733;
//   float k2 = 0.019439878275353862;
//   float k3 = -0.03544505860721041;
//   float k4 = 0.022121647569599227;

//   // 内参
//   float fx = 1059.6087870662222;
//   float fy = 1059.5973705610731;
//   float cx = 1050.3552896004385;
//   float cy = 731.3912426749002;


// };


class RadtanDistortModel : public DistortModel
{

    public:

    RadtanDistortModel(int width, int height, int wOrg, int hOrg, std::vector<float> original_intrinsics, std::vector<float> distortion_coeffs);

    void distortCoordinates(float* in_u, float* in_v, int n, float in_fx, float in_fy, float in_cx, float in_cy,
    float* out_u, float* out_v) override;

    protected:

    // distortion parameters
    float k1_org_;
    float k2_org_;
    float p1_org_;
    float p2_org_;

    // original intrinsics
    float fx_org_;
    float fy_org_;
    float cx_org_;
    float cy_org_;



};

}


