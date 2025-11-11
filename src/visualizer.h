#pragma once
#include <matplot/matplot.h>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>

#include "datasets/dataset.h"
#include "configs.h"

namespace MVS
{



class Visualizer
{   
    public:
    Visualizer(Config* const config);
    virtual ~Visualizer()  = default;
    void setDataset(Dataset* const dataset);
    void saveDepthToPCD(const PixelPoint* const ptr_depth, int width, int height, const Eigen::Matrix3f &K,
                    const std::string& save_path);
    void saveDepthToPCD(const PixelPoint* const ptr_depth, const cv::Mat &rgb, int width, int height, const Eigen::Matrix3f &K,
                    const std::string& save_path);
    void saveDepth(const Image* const image);
    static void ColorizedCVDepth(const PixelPoint* const ptr_pixel_point_matrix, int width, int height, float min_depth, float max_depth, cv::Mat& colorized_depth);
    static void showColorizedMinMaxDiffDepth(const float* const min_depth_ptr, const float* const max_depth_ptr, int width, int height, float min_depth, float max_depth, const std::string& name);
    static void showGrayImage(const float* const ptr_gray_data, int width, int height,  const std::string &name);   

    virtual void showRefImageReconstruction(Image* const ref_image, Image* const tar_image) = 0;


    protected:
    Config* config_;
    Dataset* dataset_;

    int image_window_width_ = 360;
    int image_window_height_ = 360;




};
    

    
}