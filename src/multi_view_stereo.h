#pragma once


#include <limits>
#include <matplot/matplot.h>
#include <chrono>

#include "configs.h"
#include "datasets/dataset.h"
#include "utils.h"
#include "iridescence_visualizer.h"


namespace MVS
{


class MultiViewStereo
{
  
public:
    MultiViewStereo(Config* const config);
    ~MultiViewStereo();
    void setDataset(Dataset* const dataset);
    void setVisualizer(Visualizer* const visualizer);
    void setReferenceImage(Image* const image);
    void run();
    void match(Image* const ref_image, Image* const tar_image, bool debug_plot = false);
    void getValidTarUV(Eigen::Vector2f& valid_tar_uv, const Eigen::Vector2f tar_uv, const std::vector<Eigen::Vector2f>& intersections_in_tar_image);
    bool epipolarSearch(PixelPoint* const ref_ptr_pixel_point_matrix, uint32_t ref_u, uint32_t ref_v, uint32_t min_max_idx, const Image* const tar_image, 
    Eigen::Vector2f &tar_uv_min, Eigen::Vector2f &tar_uv_max, Eigen::Vector2f& tar_uv_best_match, bool debug_plot = false);

    float SAD(const PixelPoint* ptr_ref_pixel_point_matrix, uint32_t ref_u, uint32_t ref_v, const PixelPoint* ptr_tar_pixel_point_matrix, uint32_t tar_u, uint32_t tar_v, uint32_t width, uint32_t height, int half_ws);
    float ZSAD(const cv::Mat p1, const cv::Mat p2);
    float NCC(const cv::Mat& p1, const cv::Mat& p2);
    float Census(const cv::Mat& p1, const cv::Mat& p2);

protected:
    Config* config_;
    Image* ref_image_;
    Dataset* dataset_;
    float ACCEPTABLE_MINI_COST = 20;
    int half_ws_ = 2; // half size of match window
    // double max_inv_depth_ = 5;
    // double min_inv_depth_ = 0.02;
    Visualizer *visualizer_ = nullptr;
};






}