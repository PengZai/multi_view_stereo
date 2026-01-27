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

class MultiViewStereo;

class Visualizer
{   
    public:
    Visualizer(Config* const config);
    virtual ~Visualizer()  = default;
    void setDataset(Dataset* const dataset);
    void setMultiViewStereo(MultiViewStereo* const multi_view_stereo);
    void saveDepthToPCD(const PixelPoint* const ptr_depth, int width, int height, const Eigen::Matrix3f &K,
                    const std::string& save_path);
    void saveDepthToPCD(const PixelPoint* const ptr_depth, const cv::Mat &rgb, int width, int height, const Eigen::Matrix3f &K,
                    const std::string& save_path);
    void saveDepth(const Image* const image);
    void setSelectedRefImageId(int selected_ref_image_id);
    void setSelectedTarImageId(int selected_tar_image_id);
    static void ColorizedCVDepth(Config* config, const PixelPoint* const ptr_pixel_point_matrix, int width, int height, float min_depth, float max_depth, cv::Mat& colorized_depth);
    static void showColorizedMinMaxDiffDepth(const float* const min_depth_ptr, const float* const max_depth_ptr, int width, int height, float min_depth, float max_depth, const std::string& name);
    static void showGrayImage(const float* const ptr_gray_data, int width, int height,  const std::string &name);   

    void setIsNextAction(bool isNextAction);

    virtual void showInterface() = 0;


    protected:
    Config* config_;
    Dataset* dataset_;
    MultiViewStereo* multi_view_stereo_;

    int image_window_width_ = 360;
    int image_window_height_ = 360;

    PixelPoint* ptr_picked_ref_ptr_pixel_point_;
    PixelPoint* ptr_pixel_point_on_the_left_ref_picked_pixel_point_;
    PixelPoint* ptr_pixel_point_on_the_right_ref_picked_pixel_point_;
    PixelPoint* ptr_pixel_point_on_the_up_ref_picked_pixel_point_;
    PixelPoint* ptr_pixel_point_on_the_down_ref_picked_pixel_point_;

    int last_selected_ref_image_id_ = 0;
    int selected_ref_image_id_ = 0;
    int last_selected_tar_image_id_ = 0;
    int selected_tar_image_id_ = 0;
    
    int selected_idx_display_mode_ = 1;

    bool isShowOnConsistencycheck_ = false;

    bool isNextAction_ = false;


};
    

    
}