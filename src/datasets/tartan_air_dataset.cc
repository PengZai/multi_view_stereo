#include "tartan_air_dataset.h"



namespace MVS
{

TartanAirDataset::TartanAirDataset(Config* const config)
:Dataset(config)
{

}

void TartanAirDataset::readTrajectory()
{

    std::string path = config_->trajectory_->trajectory_path_;

    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << path << std::endl;
    }

    std::string line;
    int Nline = 0;
    int Ntraj = 0;
    while (std::getline(file, line)) 
    {
        // skip comments
        if (line.empty() || line[0] == '#'){
            Nline++;
            continue;
        }
        std::istringstream iss(line);
        float x, y, z, qx, qy, qz, qw;
        if (!(iss >> x >> y >> z >> qx >> qy >> qz >> qw)) {
            std::cout << " skip malformed lines at " << Nline << std::endl;
            Nline++;
            continue; 
        }


        Eigen::Matrix4f T_world_pose = Eigen::Matrix4f::Identity();
        T_world_pose.block<3,3>(0,0) = Eigen::Quaternionf(qw,qx,qy,qz).toRotationMatrix();
        T_world_pose.block<3,1>(0,3) = Eigen::Vector3f(x,y,z);

        // std::string str_timestamp = std::to_string(timestamp);
        for(size_t cam_id=0;cam_id<config_->num_used_camera_; cam_id++){
            Image *image = new Image(config_);
            Eigen::Matrix4f T_world_camid = T_world_pose * config_->trajectory_->T_pose_camidx_[cam_id];

            std::string image_name = config_->cameras_[cam_id].image_names_[Ntraj];
            // T_world_camera
            image->setTranslation(T_world_camid.block<3,1>(0,3));
            image->setQuaternion(Eigen::Quaternionf(T_world_camid.block<3,3>(0,0)));
            image->setName(image_name);
            image->setPath(cameras_[cam_id].dir_path_ + "/" + image_name+".png");
            image->setCameraId(cam_id);
            image->setPoseId(Ntraj);
            if(config_->is_use_GT_depth_ == true){
                // std::string gt_depth_path = config_->cameras_[cam_id].gt_depth_->path_ + "/" + image->getImageName() + "_depth" + ".npy";
                std::string gt_depth_path = config_->cameras_[cam_id].gt_depth_->path_ + "/" + image->getImageName() + "_depth" + ".tiff";

                image->setGTDepthPath(gt_depth_path);    
            }

            images_.push_back(image);
        }

        Nline++;
        Ntraj++;
        if(maximum_traj_ > 0 && Ntraj >= maximum_traj_){
            break;
        }
        

    }

}


// bool TartanAirDataset::loadGTDepth(Image* const image)
// {
//     std::string path = image->getGTDepthPath();
//     NpyArray arr = loadNpy(path);
//     if (arr.shape.size() != 2)
//         throw std::runtime_error("Expected 2D array");

//     int rows = static_cast<int>(arr.shape[0]);
//     int cols = static_cast<int>(arr.shape[1]);

//     cv::Mat gt_depth;

//     if (arr.dtype == "float32" || arr.dtype == "f4") {
//         // Already float32 use directly
//         cv::Mat depth(rows, cols, CV_32F, (void*)arr.raw_data.data());
//         depth.setTo(0, depth >= 10000);

//         gt_depth = depth;
//     } 
//     else if (arr.dtype == "uint16" || arr.dtype == "u2") {
//         // Convert from uint16 float32
//         cv::Mat depth16(rows, cols, CV_16U, (void*)arr.raw_data.data());
//         depth16.setTo(0, depth16 >= 10000);
//         depth16.convertTo(gt_depth, CV_32F, 1.0f / 1000.0f); // mm → meters
//     } 
//     else {
//         throw std::runtime_error("Unsupported dtype: " + arr.dtype);
//     }

//     image->setGTDepth(gt_depth);

//     // cv::Mat depth_normalized;
//     // cv::normalize(gt_depth, depth_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);

//     // cv::Mat depth_colormap;
//     // cv::applyColorMap(depth_normalized, depth_colormap, cv::COLORMAP_JET);
//     // cv::imshow("Depth Map", depth_colormap);
//     // cv::waitKey(0);

//     return true;
// }
    
    
} // namespace MVS
