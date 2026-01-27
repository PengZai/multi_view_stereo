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

    enum Status {
        FREE = 0,
        BUSY = 1,
    };

    static inline std::string StatusToString(Status v)
    {
        switch (v)
        {
            case FREE:   return "FREE";
            case BUSY:   return "BUSY";
            default:      return "[Unknown OS_type]";
        }
    }

    MultiViewStereo(Config* const config);
    ~MultiViewStereo();
    void setDataset(Dataset* const dataset);
    void setVisualizer(Visualizer* const visualizer);
    void setReferenceImage(Image* const image);
    Image* getReferenceImage();
    void setTargetImage(Image* const image);
    Image* getTargetImage();

    void setIsRunSingleDepthReconstruction(bool isRunSingleDepthReconstruction);
    void setIsRunSceneReconstruction(bool isRunSceneReconstruction);

    Status getStatus();

    void run();
    void single_depth_reconstruction();
    void scene_reconstruction();
    void scene_reconstruction_init();
    void match(Image* const ref_image, Image* const tar_image);
    void cost_aggregation(Image* const ref_image);

    void init_aggregate_pixel(PixelPoint& pixel_point, int direction_num);
  
    void update_uncertainty(Image* const ref_image);
    void consistency_check();

    void getValidTarUV(Eigen::Vector2f& valid_tar_uv, const Eigen::Vector2f tar_uv, const std::vector<Eigen::Vector2f>& intersections_in_tar_image);
    bool epipolarSearch(Image* const ref_image, uint32_t ref_u, uint32_t ref_v, const Image* const tar_image, 
    Eigen::Vector2f &tar_uv_min, Eigen::Vector2f &tar_uv_max);

    void getImageWithPoseIdAndCameraId(Image*& ptr_image, int inp_pose_id, int inp_camera_id);


protected:



    void cost_aggregation_left_right(Image* const ref_image, bool is_forward);
    void cost_aggregation_up_down(Image* const ref_image, bool is_forward);

    // direction: 0, right; 1 left; 2 down; 3 up
    void cost_aggregation_from_last_pixel(PixelPoint& pixel_point_in_last_pos, PixelPoint& pixel_point, int direction_num);
    void pixel_depth_estimation(float& min_inv_depth, float& max_inv_depth, const Eigen::Vector3f& Kt, const Eigen::Vector3f& KRKi_uv_homo, const Eigen::Vector2f& uv_best_match_minus, const Eigen::Vector2f& uv_best_match_plus);

    Config* config_;
    Image* ref_image_;
    Image* tar_image_;
    Dataset* dataset_;
    float ACCEPTABLE_MINI_COST = 30;
    float ACCEPTABLE_COST_DIFF = 4;
    float ACCEPTABLE_DEPTH_PARAMETER = 0.05;
    float MAXIMUM_AGGREAGTE_COST_PENALTY = 6;
    float MAXIMUM_RECONSTRUCTION_DISTANCE = 0.8;
     
    uint8_t half_ws_ = 2; // half size of match window
    // double max_inv_depth_ = 5;
    // double min_inv_depth_ = 0.02;

    bool isRunSingleDepthReconstruction_ = false;
    bool isRunSceneReconstruction_ = false;

    Status status_;
    
    Visualizer *visualizer_ = nullptr;


};






}