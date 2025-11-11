#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <pangolin/pangolin.h>

#include "../src/configs.h"
#include "../src/datasets/dataset.h"
#include "../src/utils.h"
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
    MVS::Image* first_image = nullptr;

    for(size_t ref_image_idx=0; ref_image_idx < images.size(); ref_image_idx++){
        MVS::Image* ref_image = images[ref_image_idx];

        int ref_pose_id = ref_image->getPoseId();
        int ref_camera_id = ref_image->getCameraId();

        if(ref_pose_id == config->ref_pose_idx_ && ref_camera_id == config->ref_camera_idx_)
        {
            first_image = images[ref_image_idx];
            break;
        }
    }

    if(first_image == nullptr){
        std::cout << "ref_image is not existed for ref_pose_id:" << config->ref_pose_idx_ << " and ref_camera_id:" << config->ref_camera_idx_ << std::endl;
        return -1;
    }

    first_image->loadData();
    // visualizer->showUndistortedGrayImage(first_image, first_image->getImageName());
    // cv::waitKey(0);

    Eigen::Matrix4f T_first_camera_world = MVS::invertTransform(first_image->getTransformationMatrix());


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
            Eigen::Matrix4f T_world_camera = image->getTransformationMatrix();
            if(image_pose_id == first_image->getPoseId() && cam_id == first_image->getCameraId()){
                Eigen::Vector4i color_rgba = Eigen::Vector4i(0,255,0,255);
                visualizer->drawFrame(T_world_camera, color_rgba, true, std::to_string(image->getPoseId()));

            }
            else if(i==images.size()-1){
                Eigen::Vector4i color_rgba = Eigen::Vector4i(255,0,0,255);
                visualizer->drawFrame(T_world_camera, color_rgba, true, std::to_string(image->getPoseId()));
            }
            else{
                Eigen::Vector4i color_rgba = Eigen::Vector4i(0,0,255,25);
                visualizer->drawFrame(T_world_camera, color_rgba, false, std::to_string(image->getPoseId()));
            }
            // drawFrame(T_world_camera, color, true, std::to_string(image->getId()));

        }
        
        pangolin::glDrawAxis(0.5);
        pangolin::FinishFrame();
    }
    return 0;
}