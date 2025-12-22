#include "visualizer.h"

namespace MVS{

Visualizer::Visualizer(Config* const config)
{

    config_ = config;
    image_window_width_ = config_->cameras_[0].resolution_[0];
    image_window_height_ = config_->cameras_[0].resolution_[1];

    float viewer_width = config_->cameras_[0].resolution_[0];
    float viewer_height = config_->cameras_[0].resolution_[1];

}



void Visualizer::setDataset(Dataset* const dataset)
{
    dataset_ = dataset;
}

// Save depth image as point cloud in PLY format
void Visualizer::saveDepthToPCD(const PixelPoint* const ptr_pixel_point_matrix, int width, int height, const Eigen::Matrix3f &K,
                    const std::string& save_path) 
{


    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    float fx = K(0,0);
    float fy = K(1,1);
    float cx = K(0,2);
    float cy = K(1,2);
    
    std::vector<cv::Point3f> points;
    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
            float z = ptr_pixel_point_matrix[v*width + u].depth_; // depth in meters
            if (z <= 0 || std::isnan(z)) continue;

            float x = (u - cx) * z / fx;
            float y = (v - cy) * z / fy;
            cloud->points.emplace_back(x, y, z);
        }
    }

    // Write PLY header
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;

    pcl::io::savePCDFileBinary(save_path, *cloud);

    std::cout << "Saved point cloud: " << save_path << " with " 
              << cloud->points.size() << " points.\n";

}

void Visualizer::saveDepthToPCD(const PixelPoint* const ptr_pixel_point_matrix, const cv::Mat &rgb, int width, int height, const Eigen::Matrix3f &K,
                    const std::string& save_path) 
{


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    float fx = K(0,0);
    float fy = K(1,1);
    float cx = K(0,2);
    float cy = K(1,2);
    
    std::vector<cv::Point3f> points;
    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
            float z = ptr_pixel_point_matrix[v*width + u].depth_; // depth in meters
            if (z <= 0 || std::isnan(z)) continue;

            float x = (u - cx) * z / fx;
            float y = (v - cy) * z / fy;

            pcl::PointXYZRGB p;
            p.x = x;
            p.y = y;
            p.z = z;

            // read color (OpenCV stores BGR)
            const cv::Vec3b& color = rgb.at<cv::Vec3b>(v, u);
            p.b = color[0];
            p.g = color[1];
            p.r = color[2];


            cloud->points.emplace_back(p);
        }
    }

    // Write PLY header
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;

    pcl::io::savePCDFileBinary(save_path, *cloud);

    std::cout << "Saved point cloud: " << save_path << " with " 
              << cloud->points.size() << " points.\n";

}




void Visualizer::showColorizedMinMaxDiffDepth(const float* const min_depth_ptr, const float* const max_depth_ptr, int width, int height, float min_depth, float max_depth, const std::string& name)
{

    float acceptable_maximum_diff_depth = 1e-8;
    int wh = width * height;
        
    for(int i=0;i<wh;i++)
    {
        float est_min_depth = min_depth_ptr[i];
        float est_max_depth = max_depth_ptr[i];
        float d = est_max_depth - est_min_depth;

        if(d > acceptable_maximum_diff_depth)
        {
            acceptable_maximum_diff_depth = d;
        }
        

    }

    cv::Mat colorized_diffdepth(height, width, CV_8UC3);

    for(int v=0;v<height;v++)
    {
        for(int u=0;u<width;u++){
            
            float est_min_depth = min_depth_ptr[v*width+u];
            float est_max_depth = max_depth_ptr[v*width+u];
            float d = est_max_depth - est_min_depth;
            float norm_d = d / acceptable_maximum_diff_depth;

            if(est_max_depth != 0)
            {
                uchar red  = static_cast<uchar>((1.0f - norm_d) * 255);
                uchar blue   = static_cast<uchar>(norm_d * 255);
                uchar green = static_cast<uchar>((1.0f - std::abs(norm_d - 0.5f) * 2) * 255);
                colorized_diffdepth.at<cv::Vec3b>(v, u) = cv::Vec3b(blue, green, red);
            }
            else
            {
                colorized_diffdepth.at<cv::Vec3b>(v, u) = cv::Vec3b(0,0,0);
            }


        }
    }

    cv::imshow(name, colorized_diffdepth);

}

void Visualizer::ColorizedCVDepth(const PixelPoint* const ptr_pixel_point_matrix, int width, int height, float min_depth, float max_depth, cv::Mat& colorized_depth)
{

    float acceptable_maximum_depth = 0;
    int wh = width * height;
    
    for(int i=0;i<wh;i++)
    {
        float d = ptr_pixel_point_matrix[i].depth_;
        if(d >= min_depth && d <= max_depth){

            if(d > acceptable_maximum_depth)
            {
                acceptable_maximum_depth = d;
            }
        }

    }


    for(int v=0;v<height;v++)
    {
        for(int u=0;u<width;u++){
            
            float d = ptr_pixel_point_matrix[v*width+u].depth_;
            if(d >= min_depth && d <= max_depth && d <= acceptable_maximum_depth){

                float norm_d = d / acceptable_maximum_depth;
                uchar red  = static_cast<uchar>((1.0f - norm_d) * 255);
                uchar blue   = static_cast<uchar>(norm_d * 255);
                uchar green = static_cast<uchar>((1.0f - std::abs(norm_d - 0.5f) * 2) * 255);
                colorized_depth.at<cv::Vec3b>(v, u) = cv::Vec3b(blue, green, red);

            }
            else{
                colorized_depth.at<cv::Vec3b>(v, u) = cv::Vec3b(0,0,0);
            }


        }
    }


    // cv::Mat depth(height, width, CV_32FC1);
    // cv::Mat depthNormalized;
    // float minVal = std::numeric_limits<float>::infinity();
    // float maxVal = 0;

    // for(int v=0;v<height;v++)
    // {
    //     for(int u=0;u<width;u++){

    //         float d = ptr_pixel_point_matrix[v*width+u].depth_;
    //         if( d < minVal){
    //             minVal = d;
    //         }
    //         if(d > maxVal){
    //             maxVal = d;
    //         }
    //         depth.at<float>(v, u) = d;

    //     }
    // }

    
    // depth.convertTo(depthNormalized, CV_8UC1, 255.0 / (maxVal - minVal), -minVal);

    // // Apply a colormap (optional)
    // cv::applyColorMap(depthNormalized, colorized_depth, cv::COLORMAP_JET);

}




void Visualizer::saveDepth(const Image* const image)
{
    cv::Mat depth = image->getCVDepth();
    PixelPoint* pixel_point = image->getPixelPointMatrixPtr();
    int width = image->getWidth();
    int height = image->getHeight();


    cv::Mat depth_normalized;
    cv::normalize(depth, depth_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat depth_colormap;
    cv::applyColorMap(depth_normalized, depth_colormap, cv::COLORMAP_JET);

    const Eigen::Matrix3f K = config_->cameras_[image->getCameraId()].getIntrinsicsMatrix();

    const cv::Mat &rgb = image->getBGRData();
    if(!rgb.empty()){
        saveDepthToPCD(pixel_point, rgb, width, height, K, config_->save_figure_path_+"/depth_map.pcd");
    }
    else{
        saveDepthToPCD(pixel_point, width, height, K, config_->save_figure_path_+"/depth_map.pcd");
    }
    cv::imwrite(config_->save_figure_path_+"/vis_depth_map.png", depth_colormap);
    cv::imwrite(config_->save_figure_path_+"/depth_map.tiff", depth);
    // cv::imshow("Depth Map", depth_colormap);


}


void Visualizer::showGrayImage(const float* const ptr_gray_data, int width, int height,  const std::string &name)
{

    cv::Mat vis_gray(height, width, CV_8UC1);
    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
            uint8_t pixel_value = (uint8_t)ptr_gray_data[v*width+u];
            vis_gray.at<uchar>(v, u) = pixel_value;
        }
    }

    cv::imshow(name, vis_gray);
}


}