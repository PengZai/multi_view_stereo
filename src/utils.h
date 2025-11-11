#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
// #include <regex>




namespace MVS
{



// struct NpyArray {
//     std::vector<size_t> shape;
//     std::vector<unsigned char> raw_data;
//     std::string dtype; // e.g. "float32" or "uint16"
// };


// NpyArray loadNpy(const std::string& path);

float getBilinearInterpolated(const cv::Mat& img, float u, float v);
float getBilinearInterpolated(
    const float* gray, int width, int height,
    float u, float v);
float getBilinearInterpolated(
    const uint8_t* gray, int width, int height,
    float u, float v);
void getBilinearInterpolatedGradient(
    const float* grad, int width, int height,
    float u, float v,
    float& gx, float& gy);
cv::Mat getSubpixelPatch(const cv::Mat img, float u, float v,
                         int width, int height);
void getRoundPixelPatch(const cv::Mat& img, float u, float v,
                         int width, int height, cv::Mat& out_patch);
Eigen::Matrix4f invertTransform(const Eigen::Matrix4f& T);
    
} // namespace MVS






