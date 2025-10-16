#include "visualizer.h"




namespace MVS
{

Visualizer::Visualizer(Config* const config)
{
    config_ = config;
    float viewer_width = 1080;
    float viewer_height = 720;

    pangolin::CreateWindowAndBind("Pangolin Viewer", viewer_width, viewer_height);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

   s_cam_ =  pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(viewer_width,viewer_height,500,500,512,389,0.1,1000),
        pangolin::ModelViewLookAt(
        0.0, -0.7, -1.8,   // move eye closer
        0, 0, 0,
        0.0, -1.0, 0.0
    )
    );

    d_cam_ = pangolin::CreateDisplay()
        .SetBounds(0.0,1.0,0.0,1.0,-viewer_width/viewer_height)
        .SetHandler(new pangolin::Handler3D(this->s_cam_));


}

Visualizer::~Visualizer()
{

}


// Save depth image as point cloud in PLY format
void Visualizer::saveDepthToPCD(const float* const ptr_depth, int width, int height, const Eigen::Matrix3f &K,
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
            float z = ptr_depth[v*width + u]; // depth in meters
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

void Visualizer::saveDepthToPCD(const float* const ptr_depth, const cv::Mat &rgb, int width, int height, const Eigen::Matrix3f &K,
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
            float z = ptr_depth[v*width + u]; // depth in meters
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

void Visualizer::showDepth(const Image* const image)
{
    const float* const ptr_depth = image->getDepthPtr();
    int width = image->getWidth();
    int height = image->getHeight();

    cv::Mat depth(height, width, CV_32F, (void*)ptr_depth);
    cv::Mat depth_normalized;
    cv::normalize(depth, depth_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat depth_colormap;
    cv::applyColorMap(depth_normalized, depth_colormap, cv::COLORMAP_JET);

    const Eigen::Matrix3f K = config_->cameras_[image->getCameraId()].getIntrinsicsMatrix();

    const cv::Mat &rgb = image->getRGBData();
    if(!rgb.empty()){
        saveDepthToPCD(ptr_depth, rgb, width, height, K, config_->save_figure_path_+"/depth_map.pcd");
    }
    else{
        saveDepthToPCD(ptr_depth, width, height, K, config_->save_figure_path_+"/depth_map.pcd");
    }
    cv::imwrite(config_->save_figure_path_+"/vis_depth_map.png", depth_colormap);
    cv::imwrite(config_->save_figure_path_+"/depth_map.tiff", depth);
    cv::imshow("Depth Map", depth_colormap);


}


void Visualizer::showUndistortedGrayImage(const Image* const image, const std::string &name)
{

    const float* const ptr_gray_data = image->getGrayDataPtr();
    int width = image->getWidth();
    int height = image->getHeight();
    cv::Mat vis_gray(height, width, CV_8UC1);
    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
            uint8_t pixel_value = (uint8_t)ptr_gray_data[v*width+u];
            vis_gray.at<uchar>(v, u) = pixel_value;
        }
    }

    cv::imshow(name, vis_gray);
}


pangolin::View& Visualizer::getPangolinViewer()
{
    return d_cam_;
}

pangolin::OpenGlRenderState& Visualizer::getPangolineRenderState()
{
    return s_cam_;
}


void Visualizer::drawFrame(const Eigen::Matrix4f &T_w_c, const Eigen::Vector3i &bgr, bool drawAxis, const std::string &text)
{


    const float w = 0.1;
    const float h = w;
    const float z = 2*w;
    const float frame_line_width = 2.0;


    glPushMatrix();

    glMultMatrixf((GLfloat*)T_w_c.data());

    glPointSize(10.0f);  // Set point size in pixels
    glBegin(GL_POINTS);
    glColor3f(bgr[2]/255.0f,bgr[1]/255.0f,bgr[0]/255.0f);   // Set point color (red)
    glVertex3f(0, 0, 0);  // Plot point at (x=0, y=0, z=0)
    glEnd();

    // Draw axis, red - x green - y blue -z
    if(drawAxis){
        pangolin::glDrawAxis(w);
    }

    glLineWidth(frame_line_width);
    glColor3d(bgr[2]/255.0f,bgr[1]/255.0f,bgr[0]/255.0f);
    glBegin(GL_LINES);

    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);
    
    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);
    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);
    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);
    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glColor4f(bgr[2]/255.0f,bgr[1]/255.0f,bgr[0]/255.0f, 0.3f); // RGBA
    glBegin(GL_QUADS);
    glVertex3f(-w, -h, z); // bottom-left
    glVertex3f(w, -h, z);  // bottom-right
    glVertex3f(w, h, z);   // top-right
    glVertex3f(-w, h, z);  // top-left
    glEnd();



    // 🔹 Draw text label
    pangolin::GlFont& font = pangolin::GlFont::I();
    glColor3f(1.0f, 0.0f, 0.0f); // red text
    font.Text(text).Draw(0, 0, 0.02);  // slightly in front of frame

    glPopMatrix();

    glEnd();

}


void Visualizer::drawPoint(const Eigen::Vector3f &pt3f, const Eigen::Vector3i &bgr){

    float point_size = 2.0;
    glPointSize(point_size);
    glBegin(GL_POINTS);
    glVertex3f(pt3f[0],pt3f[1],pt3f[2]);
    glColor3d(bgr[2]/255.0f,bgr[1]/255.0f,bgr[0]/255.0f);
    glEnd();

}

void Visualizer::drawPoint(const Eigen::Vector3f &pt3f, float gray)
{
    float point_size = 2.0f;
    glPointSize(point_size);

    // Normalize intensity if necessary
    float c = gray;                // could be 0–255 or 0–1
    if (c > 1.0f) c /= 255.0f;     // normalize to [0,1]
    c = std::clamp(c, 0.0f, 1.0f); // ensure valid range

    // Draw point
    glBegin(GL_POINTS);
    glColor3f(c, c, c);            // same intensity for R, G, B
    glVertex3f(pt3f[0], pt3f[1], pt3f[2]);
    glEnd();
}


} // namespace MVS



