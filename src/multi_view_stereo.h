#pragma once


#include <limits>
#include <matplot/matplot.h>
#include <chrono>

#include "configs.h"
#include "data.h"
#include "utils.h"


namespace MVS
{


class MultiViewStereo
{
  
public:
    MultiViewStereo(Config* const config);
    ~MultiViewStereo();
    void setDataset(Dataset* const dataset);
    void setReferenceImage(Image* const image);
    void run();
    void match(Image* const ref_image, Image* const tar_image, bool debug_plot = false);
    void epipolarSearch(const cv::Mat ref_patch, const Image* const tar_image, 
        const Eigen::Vector2f &uv_min, const Eigen::Vector2f &uv_max, Eigen::Vector2f uv_best_match, bool debug_plot = false);

    float SAD(const cv::Mat p1, const cv::Mat p2);
    float ZSAD(const cv::Mat p1, const cv::Mat p2);
    float NCC(const cv::Mat& p1, const cv::Mat& p2);
    float Census(const cv::Mat& p1, const cv::Mat& p2);

private:
    Config* config_;
    Image* ref_image_;
    Dataset* dataset_;
    double ws_ = 5; // size of match window
    double max_inv_depth_ = 5;
    double min_inv_depth_ = 0.02;
};






}