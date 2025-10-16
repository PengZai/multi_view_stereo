#include "dataset.h"
#include "botanic_gardent_dataset.h"
#include "fast_livo2_dataset.h"
#include "remode_dataset.h"
#include "tartan_air_dataset.h"
#include "kitti_dataset.h"
#include "virtual_kitti_dataset.h"

namespace MVS
{

int Image::Nimage_ = 0;




Dataset* getDataset(Config* const config)
{

    Dataset* dataset = nullptr;
    if(config->name_ == "BotanicGarden")
    {
         dataset = new BotanicGardenDataset(config);
    }
    else if(config->name_ == "FASTLIVO2")
    {
         dataset = new FastLivo2Dataset(config);
    }
    else if(config->name_ == "Remode")
    {
         dataset = new RemodeDataset(config);
    }
    else if(config->name_ == "TartanAir")
    {
         dataset = new TartanAirDataset(config);
    }
    else if(config->name_ == "Kitti")
    {
         dataset = new KittiDataset(config);
    }
    else if(config->name_ == "VirtualKitti")
    {
         dataset = new VirtualKittiDataset(config);
    }
    
    dataset->readTrajectory();

    return dataset;

}



Dataset::Dataset(Config* const config)
{

    config_ = config;
    cameras_ = config_->cameras_;
    maximum_traj_ = config_->maximum_traj_;


}

Dataset::~Dataset(){
    for (auto* p : images_) delete p;
    images_.clear();
}



std::vector<Image*>& Dataset::getImages()
{
    return images_;
}


Image::Image(Config* const config)
:ptr_raw_gray_data_(nullptr),
ptr_gray_data_(nullptr),
ptr_depth_data_(nullptr),
ptr_min_inv_depth_data_(nullptr),
ptr_max_inv_depth_data_(nullptr),
GT_depth_path_(""),
config_(config)
{   
    id_ = Nimage_;
    Nimage_++ ;
}


Image::~Image(){
    delete[] ptr_raw_gray_data_;
    delete[] ptr_gray_data_;
    delete[] ptr_depth_data_;
    delete[] ptr_min_inv_depth_data_;
    delete[] ptr_max_inv_depth_data_;
    delete[] ptr_gradient_gray_data_;

}

const Eigen::Vector3f & Image::getTranslation() const
{
    return t_;
}

void Image::setTranslation(const Eigen::Vector3f &translation)
{
    t_ = translation;
}

const Eigen::Quaternionf& Image::getQuaternion() const
{
    return q_;
}

void Image::setQuaternion(const Eigen::Quaternionf &quaternion)
{
    q_ = quaternion;
}

const Eigen::Matrix3f Image::getRotationMatrix() const
{
    Eigen::Matrix3f R = q_.toRotationMatrix();
    return R;
}

const Eigen::Matrix4f Image::getTransformationMatrix() const
{
    // T_world_camera
    Eigen::Matrix4f T_world_camera = Eigen::Matrix4f::Identity();
    T_world_camera.block<3,3>(0,0) = q_.toRotationMatrix();
    T_world_camera.block<3,1>(0,3) = t_;

    return T_world_camera;
}


void Image::setName(const std::string &name)
{
    name_ = name;
}

std::string Image::getImageName() const
{
    return name_;
}


void Image::setPath(const std::string &path)
{
    path_ = path;
}

void Image::setGTDepthPath(const std::string &path)
{
    GT_depth_path_ = path;
}

void Image::setGTDepth(const cv::Mat& cv_gt_depth_data)
{
    cv_gt_depth_data_ = cv_gt_depth_data;
}

void Image::setPoseId(const int pose_id)
{
    pose_id_ = pose_id;
}

int Image::getId() const
{
    return id_;
}

void Image::setCameraId(const int camera_id)
{
    camera_id_ = camera_id;
}

int Image::getCameraId() const
{
    return camera_id_;
}

int Image::getPoseId() const
{
    return pose_id_;
}

void Image::setTimestamp(const double timestamp)
{
    timestamp_ = timestamp;
}

bool Image::loadData()
{
    if(path_.empty())
    {
        std::cout <<  "path of image id :" << id_ << " is empty " << std::endl;
    }

    cv::Mat cv_raw_gray_data;
    if(ptr_raw_gray_data_ != nullptr)
    {
        return true;
    }

    cv_rgb_data_ = cv::imread(path_, cv::IMREAD_COLOR);
    if (!cv_rgb_data_.empty()) {
        // Successfully read as RGB
        cv::cvtColor(cv_rgb_data_, cv_raw_gray_data, cv::COLOR_BGR2GRAY);
    } else {
        // Failed to read RGB, try grayscale
        cv_raw_gray_data = cv::imread(path_, cv::IMREAD_GRAYSCALE);
        if (cv_raw_gray_data.empty()) {
            std::cout << "Error: Failed to load image from " << path_ << std::endl;
            return false;
        }
    }

    width_ = cv_raw_gray_data.cols;
    height_ = cv_raw_gray_data.rows;
    


    ptr_raw_gray_data_ = new uint8_t[width_ * height_]();
    memcpy(ptr_raw_gray_data_, cv_raw_gray_data.data, width_ * height_);


    ptr_gray_data_ = new float[width_ * height_]();
    config_->cameras_[camera_id_].undistort_->run(this);

    ptr_depth_data_ = new float[width_ * height_]();
    ptr_min_inv_depth_data_ = new float[width_ * height_]();
    ptr_max_inv_depth_data_ = new float[width_ * height_]();
    
    // cv::Mat depth(height_, width_, CV_32F, (void*)ptr_depth_data_);
    // cv::Mat depth_normalized;
    // cv::normalize(depth, depth_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    // cv::imshow("depth", depth);
    // cv::waitKey(0);

    std::fill(ptr_min_inv_depth_data_, ptr_min_inv_depth_data_ + width_ * height_, 1.0/config_->max_depth_);
    std::fill(ptr_max_inv_depth_data_, ptr_max_inv_depth_data_ + width_ * height_, 1.0/config_->min_depth_);

    ptr_gradient_gray_data_ = new float[2 * width_ * height_]();

    makeGraident();

    return true;

}

void Image::makeGraident()
{

    // make gradient
    for (int y = 1; y < height_ - 1; ++y)
    {
        for (int x = 1; x < width_ - 1; ++x)
        {
            int idx = y * width_ + x;

            float gx = 0.5f * (ptr_gray_data_[(y) * width_ + (x + 1)] -
                            ptr_gray_data_[(y) * width_ + (x - 1)]);

            float gy = 0.5f * (ptr_gray_data_[(y + 1) * width_ + (x)] -
                            ptr_gray_data_[(y - 1) * width_ + (x)]);

            ptr_gradient_gray_data_[2 * idx + 0] = gx; // y*width_*2 + x*2 + 0
            ptr_gradient_gray_data_[2 * idx + 1] = gy; // y*width_*2 + x*2 + 1
        }
    }

}

uint32_t Image::getWidth() const
{
    return width_;
}

uint32_t Image::getHeight() const
{
    return height_;
}


uint8_t* Image::getRawGrayDataPtr() const
{
    return ptr_raw_gray_data_;
}

float* Image::getGrayDataPtr() const
{
    return ptr_gray_data_;
}

float* Image::getMinInvDepthDataPtr() const
{
    return ptr_min_inv_depth_data_;
}

float* Image::getMaxInvDepthDataPtr() const
{
    return ptr_max_inv_depth_data_;
}

float* Image::getGradientGrayDataPtr() const
{
    return ptr_gradient_gray_data_;
}

float* Image::getDepthPtr() const
{
    return ptr_depth_data_;
}

const cv::Mat& Image::getRGBData() const
{
    return cv_rgb_data_;
}

const cv::Mat& Image::getGTDepthData() const
{
    return cv_gt_depth_data_;
}


void Image::loadDepthFromMinMaxInvDepth() const
{

    int half_ws = config_->half_window_size_;
    for(size_t v=half_ws; v < height_ - half_ws; v++)
    {
        for(size_t u=half_ws; u < width_ - half_ws; u++)
        {   
            int current_coord = v*width_+u;
            // if(v>200&&u>300){
            //     int test = 1;
            // }

            float min_depth = 1.0/ptr_max_inv_depth_data_[current_coord];
            float max_depth = 1.0/ptr_min_inv_depth_data_[current_coord];
            float diff = abs(max_depth-min_depth);
            float depth = (min_depth+max_depth)/2.0;
            if(depth >= config_->min_depth_ && depth <= config_->max_depth_){
                if(diff < 10){
                    ptr_depth_data_[current_coord] = depth;
                }
                else{
                    ptr_depth_data_[current_coord] = 0.0;
                }
            }

        }
    }

}

bool Image::isInImage(float u, float v, int border) const
{
    if(ptr_gray_data_!= nullptr)
    {
        // int int_u = round(u);
        // int int_v = round(v);
        // if(int_u>=border && int_v>= border && int_u < gray_data_.cols - border && int_v < gray_data_.rows - border)
        // {
        //     return true;
        // }
        if(u >=border && v >= border && u < width_ - border - 1 && v < height_ - border - 1)
        {
            return true;
        }
    }

    return false;
}

}
