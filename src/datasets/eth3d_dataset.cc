#include "eth3d_dataset.h"



namespace MVS
{

ETH3DDataset::ETH3DDataset(Config* const config)
:Dataset(config)
{

}

void ETH3DDataset::readTrajectory()
{

    std::string path = config_->trajectory_->trajectory_path_;
    std::cout << "trajectory path : " <<  path << std::endl;

    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << path << std::endl;
    }

    std::string line;
    int Nline = 0;
    int Ntraj = 0;
    while (std::getline(file, line)) 
    {
        std::cout << "reading " << Nline << " line in trajectory file " << path << std::endl;
        // skip comments
        if (line.empty() || line[0] == '#'){
            Nline++;
            continue;
        }
        std::istringstream iss(line);
        
        std::string str_timestamp;
        float x, y, z, qx, qy, qz, qw;
        if (!(iss >> str_timestamp >> x >> y >> z >> qx >> qy >> qz >> qw)) {
            std::cout << " skip malformed lines at " << Nline << std::endl;
            Nline++;
            continue; 
        }

        if(Ntraj >= minimum_traj_)
        {
            double timestamp = std::stod(str_timestamp);
            str_timestamp.erase(std::remove(str_timestamp.begin(), str_timestamp.end(), '.'), str_timestamp.end());

            // std::string str_timestamp = std::to_string(timestamp);

            Eigen::Matrix4f T_world_pose = Eigen::Matrix4f::Identity();
            T_world_pose.block<3,3>(0,0) = Eigen::Quaternionf(qw,qx,qy,qz).toRotationMatrix();
            T_world_pose.block<3,1>(0,3) = Eigen::Vector3f(x,y,z);

            // T_world_camera
            for(size_t cam_id=0;cam_id<config_->num_used_camera_; cam_id++){
                Image *image = new Image(config_);
                Eigen::Matrix4f T_world_camid = T_world_pose * config_->trajectory_->T_pose_camidx_[cam_id];

                double min_diff = std::numeric_limits<double>::max();
                int image_idx = cameras_[cam_id].getSynchronizedImageByTimeStamp(timestamp, min_diff, config_->trajectory_->sync_time_diff_tolerance_);
                if(image_idx == -1){
                    continue;
                }

                image->setTranslation(T_world_camid.block<3,1>(0,3));
                image->setQuaternion(Eigen::Quaternionf(T_world_camid.block<3,3>(0,0)));
                image->setName(cameras_[cam_id].image_names_[image_idx]);
                image->setPath(cameras_[cam_id].dir_path_ + "/" + cameras_[cam_id].image_names_[image_idx]+".png");
                image->setTimestamp(timestamp);
                image->setCameraId(cam_id);
                image->setPoseId(Ntraj-minimum_traj_);
                image->setWidthOrg(config_->cameras_[cam_id].original_resolution_[0]);
                image->setHeightOrg(config_->cameras_[cam_id].original_resolution_[1]);

                if(config_->is_use_GT_depth_ == true){
                    std::string gt_depth_path = config_->cameras_[cam_id].gt_depth_->path_ + "/" + image->getImageName() + ".tiff";
                    image->setGTDepthPath(gt_depth_path);   
                }

                images_.push_back(image);

            }
        }

        Nline++;
        Ntraj++;
        if(maximum_traj_ > 0 && Ntraj >= maximum_traj_){
            break;
        }
        
    }

}



    
    
} // namespace MVS
