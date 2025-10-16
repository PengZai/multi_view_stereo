#pragma once
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pangolin/pangolin.h>

#include "datasets/dataset.h"
#include "configs.h"

namespace MVS
{


class Visualizer
{   
    public:
    Visualizer(Config* const config);
    ~Visualizer();
    void saveDepthToPCD(const float* const ptr_depth, int width, int height, const Eigen::Matrix3f &K,
                    const std::string& save_path);
    void saveDepthToPCD(const float* const ptr_depth, const cv::Mat &rgb, int width, int height, const Eigen::Matrix3f &K,
                    const std::string& save_path);
    void showUndistortedGrayImage(const Image* const image, const std::string &name);

    void showDepth(const Image* const image);

    pangolin::View& getPangolinViewer();
    pangolin::OpenGlRenderState& getPangolineRenderState();
    void drawFrame(const Eigen::Matrix4f &T_w_c, const Eigen::Vector3i &bgr, bool drawAxis, const std::string &text);
    void drawPoint(const Eigen::Vector3f &pt3f, const Eigen::Vector3i &bgr);
    void drawPoint(const Eigen::Vector3f &pt3f, float gray);


    protected:
    Config* config_;

    pangolin::View d_cam_;
    pangolin::OpenGlRenderState s_cam_;



};
    
} // namespace MVS

