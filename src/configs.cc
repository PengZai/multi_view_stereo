#include "configs.h"


namespace MVS
{


void parse(const cv::FileNode& node, Eigen::Matrix4f &parsed_values){
            
        for (size_t i = 0; i < node.size(); ++i) {
            cv::FileNode row = node[i];
            if (row.size() != 4) {
                std::cerr << "Error: Row " << i << " does not have 4 columns!" << std::endl;
                return;
            }
    
            for (size_t j = 0; j < row.size(); ++j) {
                parsed_values(i, j) = static_cast<double>(row[j]);
            }

        }

}

Config::Config(const std::string& config_path):
cameras_(nullptr),
trajectory_(nullptr),
gt_depth_(nullptr)
{

    cv::FileStorage* fs = new cv::FileStorage(config_path, cv::FileStorage::READ);
    if (!fs->isOpened()) {
        std::cout << "Failed to open YAML file for path " <<  config_path << "." << std::endl;
        return;
    }

    path_ = config_path;
    size_t lastSlash = config_path.find_last_of("/\\");
    size_t lastDot = config_path.find_last_of(".");
    name_ = config_path.substr(lastSlash + 1, lastDot - lastSlash - 1);

    (*fs)["system"]["num_used_camera"] >> num_used_camera_;
    (*fs)["system"]["use_external_trajectory_id"] >> use_external_trajectory_id_;
    (*fs)["system"]["is_use_GT_depth"] >> is_use_GT_depth_;
    (*fs)["system"]["use_GT_depth_id"] >> use_GT_depth_id_;
    (*fs)["system"]["maximum_traj"] >> maximum_traj_;
    (*fs)["system"]["ref_pose_idx"] >> ref_pose_idx_;
    (*fs)["system"]["ref_camera_idx"] >> ref_camera_idx_;
    (*fs)["system"]["tar_pose_start_idx"] >> tar_pose_start_idx_;
    (*fs)["system"]["tar_camera_start_idx"] >> tar_camera_start_idx_;
    (*fs)["system"]["image_sequence_mode"] >> image_sequence_mode_;
    (*fs)["system"]["min_depth"] >> min_depth_;
    (*fs)["system"]["max_depth"] >> max_depth_;
    (*fs)["system"]["start_match_uv"] >> start_match_uv_;
    (*fs)["system"]["half_window_size"] >> half_window_size_;
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

        cameras_[i].readDirectoryForImageNames(cameras_[i].dir_path_);

        cameras_[i].undistort_ = new Undistort();
        float sum_distortion_coeffs = std::accumulate(cameras_[i].distortion_coeffs_.begin(), cameras_[i].distortion_coeffs_.end(), 0.0f);

        if(sum_distortion_coeffs == 0){
            cameras_[i].undistort_->createRemapWithUndistortedModel(cameras_[i].resolution_[0], cameras_[i].resolution_[1]);
        }
        else if(cameras_[i].camera_model_ == "pinhole" && cameras_[i].distortion_model_ == "radtan")
        {
            cameras_[i].undistort_->createRemapWithPinholeAndRadtan(cameras_[i].resolution_[0], cameras_[i].resolution_[1], cameras_[i].intrinsics_, cameras_[i].distortion_coeffs_);
        }
        else if(cameras_[i].camera_model_ == "pinhole" && cameras_[i].distortion_model_ == "FOV")
        {
            cameras_[i].undistort_->createRemapWithPinholeAndFOV(cameras_[i].resolution_[0], cameras_[i].resolution_[1], cameras_[i].intrinsics_, cameras_[i].distortion_coeffs_);
        }

    }


    trajectory_ = new Trajectory();
    
    std::string external_trajectory_key = "external_trajectory"+std::to_string(use_external_trajectory_id_);
    (*fs)[external_trajectory_key]["name"] >> trajectory_->name_;
    (*fs)[external_trajectory_key]["world_coordinate"] >> trajectory_->world_coordinate_;
    (*fs)[external_trajectory_key]["trajectory_path"] >> trajectory_->trajectory_path_;
    (*fs)[external_trajectory_key]["sync_time_tolerance"] >> trajectory_->sync_time_tolerance_;


    for(size_t idx=0;idx<num_used_camera_;idx++){
            Eigen::Matrix4f T_pose_camidx;
            parse((*fs)[external_trajectory_key]["T_pose_cam"+std::to_string(idx)], T_pose_camidx);
            trajectory_->T_pose_camidx_.push_back(T_pose_camidx);
    }


    (*fs)["system"]["is_use_GT_depth"] >> is_use_GT_depth_;
    (*fs)["system"]["use_GT_depth_id"] >> use_GT_depth_id_;
    
    if(is_use_GT_depth_ == true)
    {
        gt_depth_ = new GT_depth();
        std::string use_GT_depth_key = "GT_depth"+std::to_string(use_GT_depth_id_);
        (*fs)[use_GT_depth_key]["name"] >> gt_depth_->name_;
        (*fs)[use_GT_depth_key]["path"] >> gt_depth_->path_;
        parse((*fs)[use_GT_depth_key]["T_cam0_pose"], gt_depth_->T_cam0_pose_);

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

void Camera::readDirectoryForImageNames(const std::string path)
{

    image_names_.clear();

    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            // Check if it's an image file
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tiff") {
                image_names_.push_back(entry.path().stem().string());
            }
        }
    }

    // Sort the list (optional, for deterministic order)
    std::sort(image_names_.begin(), image_names_.end());
    
}


int Camera::getSynchronizedImageByTimeStamp(const double timestamp, const double mini_time_diff)
{
    int best_idx = -1;
    long double min_diff = std::numeric_limits<double>::max();

    for(size_t i=0;i<image_names_.size();i++)
    {
        std::string name = image_names_[i];

        long double t = std::stod(name); // convert string to double
        t = t * 1e-9;
        // 2. Compare difference
        long double diff = std::abs(t - timestamp);
        if (diff < min_diff)
        {
            min_diff = diff;
            best_idx = i;
        }
    }

    if(min_diff < mini_time_diff)
    {
        return best_idx;
    }
    else{
        return -1;
    }
}


}