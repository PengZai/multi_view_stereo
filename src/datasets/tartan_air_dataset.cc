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
            images_.push_back(image);
        }

        Nline++;
        Ntraj++;
        if(maximum_traj_ > 0 && Ntraj >= maximum_traj_){
            break;
        }
        

    }

}

    
    
} // namespace MVS
