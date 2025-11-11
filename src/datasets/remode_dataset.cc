#include "remode_dataset.h"



namespace MVS
{


RemodeDataset::RemodeDataset(Config* const config)
:Dataset(config)
{

}

void RemodeDataset::readTrajectory()
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

        std::cout << "we are reading " << Nline << " line" << std::endl;
        // skip comments
        if (line.empty() || line[0] == '#'){
            Nline++;
            continue;
        }
        std::istringstream iss(line);        
        Image *image = new Image(config_);
        std::string image_name;
        float x, y, z, qx, qy, qz, qw;
        if (!(iss >> image_name >> x >> y >> z >> qx >> qy >> qz >> qw)) {
            std::cout << " skip malformed lines at " << Nline << std::endl;
            Nline++;
            continue; 
        }
        image_name = image_name.substr(0, image_name.find('.'));

        Eigen::Matrix4f T_world_pose = Eigen::Matrix4f::Identity();
        T_world_pose.block<3,3>(0,0) = Eigen::Quaternionf(qw,qx,qy,qz).toRotationMatrix();
        T_world_pose.block<3,1>(0,3) = Eigen::Vector3f(x,y,z);

        Eigen::Matrix4f T_world_camid = T_world_pose * config_->trajectory_->T_pose_camidx_[0];

        // T_world_camera
        image->setTranslation(T_world_camid.block<3,1>(0,3));
        image->setQuaternion(Eigen::Quaternionf(T_world_camid.block<3,3>(0,0)));
        image->setName(image_name);
        image->setPath(cameras_[0].dir_path_ + "/" + image_name+".png");
        image->setCameraId(0);
        image->setPoseId(Ntraj);
        if(config_->is_use_GT_depth_ == true){

            std::string gt_depth_path = config_->cameras_[0].gt_depth_->path_ + "/" + image->getImageName() + ".depth";
            image->setGTDepthPath(gt_depth_path);
            
        }
       

        images_.push_back(image);

     

        Nline++;
        Ntraj++;
        if(maximum_traj_ > 0 && Ntraj >= maximum_traj_){
            break;
        }
    }

}


bool RemodeDataset::loadGTDepth(Image* const image)
{


    std::string path = image->getGTDepthPath();
    int height = image->getHeight();
    int width = image->getWidth();

    std::ifstream depthmap_file_str(path);
    if (depthmap_file_str.is_open())
    {
        cv::Mat cv_gt_depth_data(height, width, CV_32FC1);
        float z;
        for(size_t r=0; r<height; ++r)
        {
            for(size_t c=0; c<width; ++c)
            {
                depthmap_file_str >> z;
                cv_gt_depth_data.at<float>(r, c) = z / 100.0f;
            }
        }

        image->setGTDepth(cv_gt_depth_data);

        depthmap_file_str.close();
        return true;
    }
    else
        return false;


}
    
    
} // namespace MVS


