#pragma once


#include <limits>
#include <matplot/matplot.h>
#include <chrono>

#include "configs.h"
#include "datasets/dataset.h"
#include "utils.h"
#include "persistent_homology.h"
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
    void match(Image* const ref_image, Image* const tar_image);
    void cost_aggregation(Image* const ref_image);

    void init_aggregate_pixel(PixelPoint& pixel_point);
  
    
    void update_uncertainty(Image* const ref_image);

    void getValidTarUV(Eigen::Vector2f& valid_tar_uv, const Eigen::Vector2f tar_uv, const std::vector<Eigen::Vector2f>& intersections_in_tar_image);
    bool epipolarSearch(PixelPoint* const ref_ptr_pixel_point_matrix, uint32_t ref_u, uint32_t ref_v, uint32_t epipolar_segment_idx, const Image* const tar_image, 
    Eigen::Vector2f &tar_uv_min, Eigen::Vector2f &tar_uv_max, Eigen::Vector2f& tar_uv_best_match);

    float SAD(const PixelPoint* ptr_ref_pixel_point_matrix, uint32_t ref_u, uint32_t ref_v, const PixelPoint* ptr_tar_pixel_point_matrix, uint32_t tar_u, uint32_t tar_v, uint32_t width, uint32_t height, int half_ws);
    float ZSAD(const cv::Mat p1, const cv::Mat p2);
    float NCC(const cv::Mat& p1, const cv::Mat& p2);
    float Census(const cv::Mat& p1, const cv::Mat& p2);

protected:

    void cost_aggregation_left_right(Image* const ref_image, bool is_forward);
    void cost_aggregation_up_down(Image* const ref_image, bool is_forward);
    bool cost_aggregation_from_last_pixel(const PixelPoint& pixel_point_in_last_pos, PixelPoint& pixel_point);
    void normalize_aggregated_cost(Image* const ref_image);
    float aggregate_cost_func(float cost_in_curr_pos, float inv_depth_in_curr_pos, float intensity_in_curr_pos, 
        float tmp_aggreagted_cost_in_last_pos,
        float mini_tmp_aggregated_cost_in_last_pos, float mini_inv_depth_in_last_pos, float intensity_in_last_pos);
    void pixel_depth_estimation(float& min_inv_depth, float& max_inv_depth, const Eigen::Vector3f& Kt, const Eigen::Vector3f& KRKi_uv_homo, const Eigen::Vector2f& uv_best_match_minus, const Eigen::Vector2f& uv_best_match_plus);


    Config* config_;
    Image* ref_image_;
    Dataset* dataset_;
    float ACCEPTABLE_MINI_COST = 20;
    float ACCEPTABLE_COST_DIFF = 2;
    float ACCEPTABLE_DEPTH_PARAMETER = 0.2;
    float MAXIMUM_AGGREAGTE_COST_PENALTY = 10;
     
    int half_ws_ = 2; // half size of match window
    // double max_inv_depth_ = 5;
    // double min_inv_depth_ = 0.02;
    Visualizer *visualizer_ = nullptr;






};






}