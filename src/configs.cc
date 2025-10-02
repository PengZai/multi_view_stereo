#include "configs.h"


namespace MVS
{

Config::Config(cv::FileStorage* fs)
{
    (*fs)["system"]["num_used_camera"] >> num_used_camera_;
    (*fs)["system"]["trajectory_path"] >> trajectory_path_;
    (*fs)["system"]["maximum_traj"] >> maximum_traj_;
    (*fs)["system"]["ref_image_idx"] >> ref_image_idx_;
    (*fs)["system"]["tar_image_start_idx"] >> tar_image_start_idx_;
    (*fs)["system"]["min_depth"] >> min_depth_;
    (*fs)["system"]["max_depth"] >> max_depth_;
    (*fs)["system"]["start_match_uv"] >> start_match_uv_;
    (*fs)["system"]["window_size"] >> window_size_;
    (*fs)["system"]["save_figure_path"] >> save_figure_path_;
    (*fs)["system"]["debug_plot"] >> debug_plot_;


    if(!std::filesystem::exists(save_figure_path_))
    {
        std::filesystem::create_directories(save_figure_path_);
    }

    cameras_ = new Camera[num_used_camera_]();

    for(int i=0; i<num_used_camera_; i++)
    {   
        std::string camKey = "cam"+std::to_string(i);
        (*fs)[camKey]["name"]  >> cameras_[i].name_;
        (*fs)[camKey]["dir_path"]  >> cameras_[i].dir_path_;
        (*fs)[camKey]["resolution"]  >> cameras_[i].resolution_;
        (*fs)[camKey]["camera_model"]  >> cameras_[i].camera_model_;
        (*fs)[camKey]["intrinsics"]  >> cameras_[i].intrinsics_;
        (*fs)[camKey]["distortion_model"]  >> cameras_[i].distortion_model_;
        (*fs)[camKey]["distortion_coeffs"]  >> cameras_[i].distortion_coeffs_;

    }

    
}


Config::~Config(){
    delete[] cameras_;
}

Camera::Camera(){

  

}


Camera::~Camera(){




}


Eigen::Matrix3f Camera::getIntrinsicsMatrix() const
{
    Eigen::Matrix3f K;
     K << this->intrinsics_[0], 0, this->intrinsics_[2],
         0, this->intrinsics_[1], this->intrinsics_[3],
         0, 0, 1;

    return K;

}


}