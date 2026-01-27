#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
// #include <regex>




namespace MVS
{

class PixelPoint;


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

float ASW(const PixelPoint* ptr_ref_pixel_point_matrix, uint32_t ref_u, uint32_t ref_v, uint32_t ref_width, const PixelPoint* ptr_tar_pixel_point_matrix, uint32_t tar_u, uint32_t tar_v, uint32_t tar_width, int half_ws);
float SAD(const PixelPoint* ptr_ref_pixel_point_matrix, uint32_t ref_u, uint32_t ref_v, uint32_t ref_width, const PixelPoint* ptr_tar_pixel_point_matrix, uint32_t tar_u, uint32_t tar_v, uint32_t tar_width, int half_ws);
float ZSAD(const PixelPoint* ptr_ref_pixel_point_matrix, uint32_t ref_u, uint32_t ref_v, uint32_t ref_width, const PixelPoint* ptr_tar_pixel_point_matrix, uint32_t tar_u, uint32_t tar_v, uint32_t tar_width, int half_ws);
float ZSAD(const PixelPoint* ptr_ref_pixel_point_matrix, uint32_t ref_u, uint32_t ref_v, float mean_ref, uint32_t ref_width, const PixelPoint* ptr_tar_pixel_point_matrix, uint32_t tar_u, uint32_t tar_v, uint32_t tar_width, int half_ws);
float ZSAD(const cv::Mat p1, const cv::Mat p2);
float NCC(const cv::Mat& p1, const cv::Mat& p2);
float Census(const cv::Mat& p1, const cv::Mat& p2);

} // namespace MVS






