#include "kitti_dataset.h"



namespace MVS
{

KittiDataset::KittiDataset(Config* const config)
:Dataset(config)
{

}

void KittiDataset::readTrajectory()
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
        float T00, T01, T02, T03, T10, T11, T12, T13, T20, T21, T22, T23;
        if (!(iss >> T00 >> T01 >> T02 >> T03 >> T10 >> T11 >> T12 >> T13 >> T20 >> T21 >> T22 >> T23)) {
            std::cout << " skip malformed lines at " << Nline << std::endl;
            Nline++;
            continue; 
        }

        if(Ntraj >= minimum_traj_)
        {
            // double timestamp = std::stod(str_timestamp);
            // str_timestamp.erase(std::remove(str_timestamp.begin(), str_timestamp.end(), '.'), str_timestamp.end());
            // std::string str_timestamp = std::to_string(timestamp);

            Eigen::Matrix4f T_world_pose;
            T_world_pose << 
            T00, T01, T02, T03, 
            T10, T11, T12, T13, 
            T20, T21, T22, T23,
            0, 0, 0, 1.0;

            // T_world_camera
            for(size_t cam_id=0;cam_id<config_->num_used_camera_; cam_id++){
                Image *image = new Image(config_);
                Eigen::Matrix4f T_world_camid = T_world_pose * config_->trajectory_->T_pose_camidx_[cam_id];
                std::string image_name = config_->cameras_[cam_id].image_names_[Ntraj];

                image->setTranslation(T_world_camid.block<3,1>(0,3));
                image->setQuaternion(Eigen::Quaternionf(T_world_camid.block<3,3>(0,0)));
                image->setName(image_name);
                image->setPath(cameras_[cam_id].dir_path_ + "/" + image_name +".png");
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
