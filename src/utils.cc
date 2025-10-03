#include "utils.h"



namespace MVS
{

float getBilinearInterpolated(const cv::Mat& img, float u, float v) 
{
    int x = floor(u);
    int y = floor(v);

    if (x < 0 || x >= img.cols-1 || y < 0 || y >= img.rows-1)
        return 0.0f; // outside image, return 0 (or handle differently)

    float dx = u - x;
    float dy = v - y;

    float I00 = img.at<uchar>(y, x);
    float I10 = img.at<uchar>(y, x+1);
    float I01 = img.at<uchar>(y+1, x);
    float I11 = img.at<uchar>(y+1, x+1);

    return (1-dx)*(1-dy)*I00 +
           dx*(1-dy)*I10 +
           (1-dx)*dy*I01 +
           dx*dy*I11;
}

cv::Mat getSubpixelPatch(const cv::Mat img, float u, float v,
                         int width, int height) 
{
    int half_w = width  / 2;
    int half_h = height / 2;

    cv::Mat patch(height, width, CV_32F);

    for (int dy = -half_h; dy <= half_h; dy++) {
        for (int dx = -half_w; dx <= half_w; dx++) {
            float uu = u + dx;
            float vv = v + dy;
            patch.at<float>(dy + half_h, dx + half_w) =
                getBilinearInterpolated(img, uu, vv);
        }
    }
    return patch;
}


void getRoundPixelPatch(const cv::Mat& img, float u, float v,
                         int width, int height, cv::Mat& out_patch) 
{
    int half_w = width  / 2;
    int half_h = height / 2;
    int round_u = round(u);
    int round_v = round(v);

    out_patch = img(cv::Rect(round_u-half_w, round_v-half_h, width, height));

}



Eigen::Matrix4f invertTransform(const Eigen::Matrix4f& T)
{   
    Eigen::Matrix3f R = T.block<3,3>(0,0);
    Eigen::Vector3f t = T.block<3,1>(0,3);

    Eigen::Matrix4f T_inv = Eigen::Matrix4f::Identity();
    T_inv.block<3,3>(0,0) = R.transpose();
    T_inv.block<3,1>(0,3) = -R.transpose() * t;
    return T_inv;
}

}

