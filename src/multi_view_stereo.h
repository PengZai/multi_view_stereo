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
    void epipolarSearch(const uint8_t* const ref_ptr_gray_data, uint32_t ref_u, uint32_t ref_v, const Image* const tar_image, 
    const Eigen::Vector2f &tar_uv_min, const Eigen::Vector2f &tar_uv_max, Eigen::Vector2f tar_uv_best_match, bool debug_plot = false);

    float SAD(const uint8_t* ref_ptr, uint32_t ref_u, uint32_t ref_v, const uint8_t* tar_ptr, uint32_t tar_u, uint32_t tar_v, uint32_t width, uint32_t height, uint32_t half_ws);
    float ZSAD(const cv::Mat p1, const cv::Mat p2);
    float NCC(const cv::Mat& p1, const cv::Mat& p2);
    float Census(const cv::Mat& p1, const cv::Mat& p2);

private:
    Config* config_;
    Image* ref_image_;
    Dataset* dataset_;
    double half_ws_ = 2; // half size of match window
    double max_inv_depth_ = 5;
    double min_inv_depth_ = 0.02;
};






}