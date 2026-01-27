#include "virtual_kitti_dataset.h"



namespace MVS
{

VirtualKittiDataset::VirtualKittiDataset(Config* const config)
:Dataset(config)
{

}

void VirtualKittiDataset::readTrajectory()
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
        if (line.empty() || line[0] == '#' || line == "frame r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0.1 0.2 1"){
            Nline++;
            continue;
        }
        std::istringstream iss(line);
        
        // frame r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0.1 0.2 1
        int frame_idx;
        float r11, r12, r13, t1, r21, r22, r23, t2, r31, r32, r33, t3, r41, r42, r43, t4;
        if (!(iss >> frame_idx >> r11 >> r12 >> r13 >> t1 >> r21 >> r22 >> r23 >> t2 >> r31 >> r32 >> r33 >> t3 >> r41 >> r42 >> r43 >> t4)) {
            std::cout << " skip malformed lines at " << Nline << std::endl;
            Nline++;
            continue; 
        }

        if(Ntraj >= minimum_traj_)
        {

            // double timestamp = std::stod(str_timestamp);
            // str_timestamp.erase(std::remove(str_timestamp.begin(), str_timestamp.end(), '.'), str_timestamp.end());
            // std::string str_timestamp = std::to_string(timestamp);

            Eigen::Matrix4f T_pose_world;
            T_pose_world << 
            r11, r12, r13, t1, 
            r21, r22, r23, t2, 
            r31, r32, r33, t3,
            0, 0, 0, 1.0;

            Eigen::Matrix4f T_world_pose = invertTransform(T_pose_world);

            // T_world_camera
            for(size_t cam_id=0;cam_id<config_->num_used_camera_; cam_id++){
                Image *image = new Image(config_);
                Eigen::Matrix4f T_world_camid = T_world_pose * config_->trajectory_->T_pose_camidx_[cam_id];
                std::string image_name = config_->cameras_[cam_id].image_names_[frame_idx];

                image->setTranslation(T_world_camid.block<3,1>(0,3));
                image->setQuaternion(Eigen::Quaternionf(T_world_camid.block<3,3>(0,0)));
                image->setName(image_name);
                image->setPath(cameras_[cam_id].dir_path_ + "/" + image_name +".png");
                image->setCameraId(cam_id);
                image->setPoseId(Ntraj-minimum_traj_);
                images_.push_back(image);
                image->setWidthOrg(config_->cameras_[cam_id].original_resolution_[0]);
                image->setHeightOrg(config_->cameras_[cam_id].original_resolution_[1]);


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
