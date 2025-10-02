#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <pangolin/pangolin.h>

#include "../src/configs.h"
#include "../src/data.h"
#include "../src/utils.h"

void drawFrame(const Eigen::Matrix4d &T_w_c, const Eigen::Vector3i &bgr, bool drawAxis){


    const float w = 0.1;
    const float h = w;
    const float z = 2*w;
    const float frame_line_width = 2.0;


    glPushMatrix();

    glMultMatrixd((GLdouble*)T_w_c.data());

    glPointSize(10.0f);  // Set point size in pixels
    glBegin(GL_POINTS);
    glColor3d(bgr[2]/255.0f,bgr[1]/255.0f,bgr[0]/255.0f);   // Set point color (red)
    glVertex3f(0, 0, 0);  // Plot point at (x=0, y=0, z=0)
    glEnd();

    // Draw axis, red - x green - y blue -z
    if(drawAxis){
        pangolin::glDrawAxis(w);
    }

    glLineWidth(frame_line_width);
    glColor3d(bgr[2]/255.0f,bgr[1]/255.0f,bgr[0]/255.0f);
    glBegin(GL_LINES);

    glVertex3d(0,0,0);
    glVertex3d(w,h,z);
    glVertex3d(0,0,0);
    glVertex3d(w,-h,z);
    glVertex3d(0,0,0);
    glVertex3d(-w,-h,z);
    glVertex3d(0,0,0);
    glVertex3d(-w,h,z);
    
    glVertex3d(w,h,z);
    glVertex3d(w,-h,z);
    glVertex3d(-w,h,z);
    glVertex3d(-w,-h,z);
    glVertex3d(-w,h,z);
    glVertex3d(w,h,z);
    glVertex3d(-w,-h,z);
    glVertex3d(w,-h,z);
    glEnd();

    glColor4f(bgr[2]/255.0f,bgr[1]/255.0f,bgr[0]/255.0f, 0.3f); // RGBA
    glBegin(GL_QUADS);
    glVertex3d(-w, -h, z); // bottom-left
    glVertex3d(w, -h, z);  // bottom-right
    glVertex3d(w, h, z);   // top-right
    glVertex3d(-w, h, z);  // top-left
    glEnd();

    glPopMatrix();

    glEnd();


}

int main(int argc, char** argv)
{
    std::string config_path = argv[1];
    cv::FileStorage* fs = new cv::FileStorage(config_path, cv::FileStorage::READ);
    if (!fs->isOpened()) {
        std::cout << "Failed to open YAML file: " << config_path << std::endl;
        return -1;
    }

    MVS::Config *config = new MVS::Config(fs);
    MVS::Dataset *dataset = new MVS::Dataset(config);

    pangolin::CreateWindowAndBind("Pangolin Viewer", 1080, 720);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1080,720,500,500,512,389,0.1,1000),
        pangolin::ModelViewLookAt(
        0.0, -0.7, -1.8,   // move eye closer
        0, 0, 0,
        0.0, -1.0, 0.0
    )
    );

    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0,1.0,0.0,1.0,-1080.0/720.0)
        .SetHandler(&handler);

    MVS::Image* first_image = dataset->images_[config->tar_image_start_idx_];
    Eigen::Matrix4d T_first_camera_world = MVS::invertTransform(first_image->getTransformationMatrix()).cast<double>();


    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f,1.0f,1.0f,1.0f);

        for(size_t i=config->tar_image_start_idx_;i < dataset->images_.size(); i++)
        {   

            MVS::Image* image = dataset->images_[i];
            Eigen::Matrix4d T_world_camera = image->getTransformationMatrix().cast<double>();
            drawFrame( T_first_camera_world * T_world_camera, Eigen::Vector3i(0,255,0), true);
        }
        
        pangolin::glDrawAxis(0.5);
        pangolin::FinishFrame();
    }
    return 0;
}