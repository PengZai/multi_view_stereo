#include "tanks_temples_dataset.h"



namespace MVS
{

TanksTemplesDataset::TanksTemplesDataset(Config* const config)
:Dataset(config)
{

}

void TanksTemplesDataset::readTrajectory()
{

    std::string path = config_->trajectory_->trajectory_path_;

    FILE * f = fopen( path.c_str(), "r" );
    if ( f == NULL ) 
    {        
        std::cerr << "Failed to open " << path << std::endl;
    }

    char buffer[1024];
    int id1, id2, frame;

    std::string line;
    int Nline = 0;
    int Ntraj = 0;
    while ( fgets( buffer, 1024, f ) != NULL )
    {
        if ( strlen( buffer ) <= 0 && buffer[ 0 ] == '#' ) {
            continue;
        }

        Eigen::Matrix4f T_world_pose;


        sscanf( buffer, "%d %d %d", &id1, &id2, &frame);
        fgets( buffer, 1024, f );
        sscanf( buffer, "%f %f %f %f", &T_world_pose(0,0), &T_world_pose(0,1), &T_world_pose(0,2), &T_world_pose(0,3) );
        fgets( buffer, 1024, f );
        sscanf( buffer, "%f %f %f %f", &T_world_pose(1,0), &T_world_pose(1,1), &T_world_pose(1,2), &T_world_pose(1,3) );
        fgets( buffer, 1024, f );
        sscanf( buffer, "%f %f %f %f", &T_world_pose(2,0), &T_world_pose(2,1), &T_world_pose(2,2), &T_world_pose(2,3) );
        fgets( buffer, 1024, f );
        sscanf( buffer, "%f %f %f %f", &T_world_pose(3,0), &T_world_pose(3,1), &T_world_pose(3,2), &T_world_pose(3,3) );

        if(Ntraj >= minimum_traj_)
        {
            // std::string str_timestamp = std::to_string(timestamp);
            for(size_t cam_id=0;cam_id<config_->num_used_camera_; cam_id++){
                Image *image = new Image(config_);
                Eigen::Matrix4f T_world_camid = T_world_pose * config_->trajectory_->T_pose_camidx_[cam_id];

                std::string image_name = config_->cameras_[cam_id].image_names_[Ntraj];
                // T_world_camera
                image->setTranslation(T_world_camid.block<3,1>(0,3));
                image->setQuaternion(Eigen::Quaternionf(T_world_camid.block<3,3>(0,0)));
                image->setName(image_name);
                image->setPath(cameras_[cam_id].dir_path_ + "/" + image_name+".jpg");
                image->setCameraId(cam_id);
                image->setPoseId(Ntraj-minimum_traj_);
                image->setWidthOrg(config_->cameras_[cam_id].original_resolution_[0]);
                image->setHeightOrg(config_->cameras_[cam_id].original_resolution_[1]);


                if(config_->is_use_GT_depth_ == true){
                    // std::string gt_depth_path = config_->cameras_[cam_id].gt_depth_->path_ + "/" + image->getImageName() + "_depth" + ".npy";
                    std::string gt_depth_path = config_->cameras_[cam_id].gt_depth_->path_ + "/" + image->getImageName() + "_depth" + ".tiff";

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
