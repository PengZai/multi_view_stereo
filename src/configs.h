#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <vector>
#include <filesystem>
#include <numeric>  // for std::accumulate
#include <cmath>

#include "undistort.h"

namespace MVS
{

class Undistort;


class Camera
{
    public:
    Camera();
    ~Camera();


    public:

    Eigen::Matrix3f getIntrinsicsMatrix() const;
    void readDirectoryForImageNames(const std::string path);
    int getSynchronizedImageByTimeStamp(const double timestamp, const double mini_time_diff);

    std::string name_;
    std::string dir_path_;
    std::vector<int> resolution_;
    std::string camera_model_;
    std::vector<float> intrinsics_;
    std::string distortion_model_;
    std::vector<float> distortion_coeffs_;
    std::vector<std::string> image_names_;
    Undistort* undistort_;
   

};

class Trajectory
{
    public:

    std::string name_;
    double sync_time_tolerance_;
    std::string world_coordinate_;
    std::string trajectory_path_;
    std::vector<Eigen::Matrix4f> T_pose_camidx_;
};


class GT_depth
{
    public:
    std::string name_;
    std::string path_;
    Eigen::Matrix4f T_cam0_pose_;


};


class Config
{
    public:
    Config(const std::string& config_path);
    ~Config();


    

    std::string path_;
    std::string name_;
    int num_used_camera_;
    std::string data_path_;
    int use_external_trajectory_id_;
    bool is_use_GT_depth_;
    int use_GT_depth_id_;
    int maximum_traj_;
    int ref_pose_idx_;
    int ref_camera_idx_;
    int tar_pose_start_idx_;
    int tar_camera_start_idx_;
    std::string image_sequence_mode_;
    float min_depth_;
    float max_depth_;
    std::vector<int> start_match_uv_;
    int half_window_size_;
    std::string save_figure_path_;
    bool debug_plot_;
    Camera* cameras_;
    Trajectory* trajectory_;
    GT_depth* gt_depth_;

};


}

