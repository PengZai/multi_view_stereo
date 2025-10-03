#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

namespace MVS
{


float getBilinearInterpolated(const cv::Mat& img, float u, float v);
cv::Mat getSubpixelPatch(const cv::Mat img, float u, float v,
                         int width, int height);
void getRoundPixelPatch(const cv::Mat& img, float u, float v,
                         int width, int height, cv::Mat& out_patch);
Eigen::Matrix4f invertTransform(const Eigen::Matrix4f& T);
    
} // namespace MVS






