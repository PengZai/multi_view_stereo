#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <vector>
#include <filesystem>

namespace MVS
{

class Camera
{
    public:
    Camera();
    ~Camera();


    public:

    Eigen::Matrix3f getIntrinsicsMatrix() const;

    std::string name_;
    std::string dir_path_;
    std::vector<int> resolution_;
    std::string camera_model_;
    std::vector<float> intrinsics_;
    std::string distortion_model_;
    std::vector<float> distortion_coeffs_;
   

};


class Config
{
    public:
    Config(cv::FileStorage* fs);
    ~Config();

    int num_used_camera_;
    std::string data_path_;
    std::string trajectory_path_;
    int maximum_traj_;
    int ref_image_idx_;
    int tar_image_start_idx_;
    float min_depth_;
    float max_depth_;
    std::vector<int> start_match_uv_;
    int window_size_;
    std::string save_figure_path_;
    bool debug_plot_;
    Camera* cameras_;

};


}

