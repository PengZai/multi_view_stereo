#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <pangolin/pangolin.h>

#include "../src/configs.h"
#include "../src/datasets/dataset.h"
#include "../src/utils.h"
#include "../src/visualizer.h"
#include "../src/pangolin_visualizer.h"



int main(int argc, char** argv)
{
    std::string config_path = argv[1];

    MVS::Config *config = new MVS::Config(config_path);

    MVS::Dataset *dataset = getDataset(config);
    MVS::PangolinVisualizer* visualizer = new MVS::PangolinVisualizer(config);



    pangolin::View& d_cam = visualizer->getPangolinViewer();
    pangolin::OpenGlRenderState& s_cam = visualizer->getPangolinRenderState(); 
        
    std::vector<MVS::Image*> images = dataset->getImages();
    MVS::Image* ref_image = nullptr;
    MVS::Image* tar_image = nullptr;

    for(size_t ref_image_idx=0; ref_image_idx < images.size(); ref_image_idx++){
        MVS::Image* image = images[ref_image_idx];

        int ref_pose_id = image->getPoseId();
        int ref_camera_id = image->getCameraId();

        if(ref_pose_id == config->ref_pose_idx_ && ref_camera_id == config->ref_camera_idx_)
        {
            ref_image = image;
            break;
        }
    }

    for(size_t tar_image_idx=0; tar_image_idx < images.size(); tar_image_idx++){
        MVS::Image* image = images[tar_image_idx];

        int tar_pose_id = image->getPoseId();
        int tar_camera_id = image->getCameraId();

        if(tar_pose_id == config->tar_pose_start_idx_ && tar_camera_id == config->tar_camera_start_idx_)
        {
            tar_image = image;
            break;
        }
    }

    if(ref_image == nullptr){
        std::cout << "ref_image is not existed for ref_pose_id:" << config->ref_pose_idx_ << " and ref_camera_id:" << config->ref_camera_idx_ << std::endl;
        return -1;
    }

    ref_image->loadData();
    if(config->is_use_GT_depth_){
        bool isSuccess = dataset->loadGTDepth(ref_image);
    }
    // visualizer->showUndistortedGrayImage(first_image, first_image->getImageName());
    // cv::waitKey(0);
    // cv::imshow("depth_truth", cv_gt_depth_data * 0.4);
    // cv::waitKey(0);

    Eigen::Matrix4f T_first_camera_world = MVS::invertTransform(ref_image->getTransformationMatrix());
    int height = ref_image->getHeight();
    int width = ref_image->getWidth();
    Eigen::Matrix3f K_cam0 = config->cameras_[0].getIntrinsicsMatrix();

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        // glClearColor(1.0f,1.0f,1.0f,1.0f);

        for(size_t i=0;i < images.size(); i++)
        {   

            MVS::Image* image = images[i];
            int image_pose_id =  image->getPoseId();
            int cam_id = image->getCameraId();
            bool isSuccess = image->loadData();   
            if(isSuccess == false){
                continue;
            }
            if(config->is_use_GT_depth_){
                isSuccess = dataset->loadGTDepth(image);
            }
            if(isSuccess == false){
                continue;
            }

            // Eigen::Matrix4f T_world_camera = T_first_camera_world * image->getTransformationMatrix();
            Eigen::Matrix4f T_world_camera = image->getTransformationMatrix();

            if(image_pose_id == ref_image->getPoseId() && cam_id == ref_image->getCameraId()){
                Eigen::Vector4i color_rgba = Eigen::Vector4i(0,255,0,255);
                visualizer->drawFrame(T_world_camera, color_rgba, true, std::to_string(image->getPoseId()));

            }
            else if(image_pose_id == tar_image->getPoseId() && cam_id == tar_image->getCameraId()){
                Eigen::Vector4i color_rgba = Eigen::Vector4i(255,0,0,255);
                visualizer->drawFrame(T_world_camera, color_rgba, true, std::to_string(image->getPoseId()));
            }
            else{
                Eigen::Vector4i color_rgba = Eigen::Vector4i(0,0,255,25);
                visualizer->drawFrame(T_world_camera, color_rgba, false, std::to_string(image->getPoseId()));
            }

            // Eigen::Matrix4f T_world_camera = image->getTransformationMatrix();

            if((image_pose_id == ref_image->getPoseId() && cam_id == ref_image->getCameraId() ) || 
                (image_pose_id == tar_image->getPoseId() && cam_id == tar_image->getCameraId() )){
            
                cv::Mat cv_gt_depth_data = image->getGTDepthData();
                if(cv_gt_depth_data.empty()){
                    std::cout << "cv_gt_depth_data is empty for " << "image_pose_id : " << image_pose_id << " and cam_id : " << cam_id << std::endl;
                    continue;
                }
                MVS::PixelPoint* ptr_pixel_point_matrix = image->getPixelPointMatrixPtr();
                for(int v=0;v<height;v++)
                {
                    for(int u=0;u<width;u++)
                    {   

                        float z = cv_gt_depth_data.at<float>(v,u);
                        if(z < config->min_depth_ || z > config->max_depth_)
                        {
                            continue;
                        }
                        float x = (u - K_cam0(0,2)) * z / K_cam0(0,0);
                        float y = (v - K_cam0(1,2)) * z / K_cam0(1,1);

                        float intensity = ptr_pixel_point_matrix[v*width+u].intensity_;   
                        Eigen::Vector4f pc_h = Eigen::Vector4f(x,y,z, 1.0f); 
                        Eigen::Vector4f pw_h = T_world_camera * pc_h;     
                        Eigen::Vector3f pw = pw_h.head<3>() / pw_h[3];   

                        visualizer->drawPoint(pw, intensity);
                        
                    }
                }
                
            }
     

            // visualizer->drawFrame( T_first_camera_world * T_world_camera, color, true, std::to_string(image->getId()));
            
            


        }
        
        pangolin::glDrawAxis(0.5);
        pangolin::FinishFrame();
    }
    return 0;
}