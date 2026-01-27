#include "dataset.h"
#include "botanic_gardent_dataset.h"
#include "fast_livo2_dataset.h"
#include "remode_dataset.h"
#include "tartan_air_dataset.h"
#include "kitti_dataset.h"
#include "tanks_temples_dataset.h"
#include "eth3d_dataset.h"
// #include "virtual_kitti_dataset.h"
#include "../undistort.h"


namespace MVS
{

int Image::Nimage_ = 0;
int PixelPoint::NPixelPoint_ = 0;



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
    else if(config->name_ == "TanksTemples")
    {
        dataset = new TanksTemplesDataset(config);
    }
    else if(config->name_ == "ETH3D")
    {
        dataset = new ETH3DDataset(config);
    }
    // else if(config->name_ == "VirtualKitti")
    // {
    //     dataset = new VirtualKittiDataset(config);
    // }
    
    dataset->readTrajectory();

    return dataset;

}

bool Dataset::loadGTDepth(Image* const image)
{

    cv::Mat cv_gt_depth_data;
    const std::string path = image->getGTDepthPath();
    cv_gt_depth_data = cv::imread(path, cv::IMREAD_UNCHANGED);
    image->setGTDepth(cv_gt_depth_data);   

    return true;

}


Dataset::Dataset(Config* const config)
{

    config_ = config;
    cameras_ = config_->cameras_;
    maximum_traj_ = config_->maximum_traj_;
    minimum_traj_ = config_->minimum_traj_;


}

Dataset::~Dataset(){
    for (auto* p : images_) delete p;
    images_.clear();
}



std::vector<Image*>& Dataset::getImages()
{
    return images_;
}



void DebugInfo::clear()
{
    // tar_pose_id_ = -1;
    // tar_camera_id_ = -1;

    steps_.clear();

    tmp_aggregate_costs_from_left_direction_.clear();
    tmp_aggregate_costs_from_right_direction_.clear();
    tmp_aggregate_costs_from_up_direction_.clear();
    tmp_aggregate_costs_from_down_direction_.clear();

    uv_min_ = Eigen::Vector2f::Zero();
    uv_max_ = Eigen::Vector2f::Zero();
    // uv_best_match_ = Eigen::Vector2f::Zero();
    unit_epipolar_vector_ = Eigen::Vector2f::Zero();
    epipolar_length_ = -1;

    init_min_depth_ = -1;
    init_max_depth_ = -1;
    init_depth_ = -1;

    uv_possible_match_ = Eigen::Vector2f::Zero();
    mini_aggregate_cost_idx_ = -1;
    mini_aggregate_cost_idx_minus_ = -1;
    mini_aggregate_cost_idx_plus_ = -1;
    updated_min_depth_ = -1;
    updated_max_depth_ = -1;
    updated_depth_ = -1;


}


void DebugInfo::appendTmpAggregateCostsFromCertainDirection(float tmp_aggregate_cost, int direction_num)
{

    if(direction_num == 0)
    {
        tmp_aggregate_costs_from_right_direction_.emplace_back(tmp_aggregate_cost);
    }
    else if(direction_num == 1)
    {
        tmp_aggregate_costs_from_left_direction_.emplace_back(tmp_aggregate_cost);
    }
    else if(direction_num == 2)
    {
        tmp_aggregate_costs_from_down_direction_.emplace_back(tmp_aggregate_cost); 

    }
    else if(direction_num == 3)
    {
        tmp_aggregate_costs_from_up_direction_.emplace_back(tmp_aggregate_cost);

    }
    


}

void DebugInfo::setZerosTmpAggregateCostsFromCertainDirection(const int N, int direction_num)
{
    
    if(direction_num == 0)
    {
        tmp_aggregate_costs_from_right_direction_.resize(N, 0);

    }
    else if(direction_num == 1)
    {
        tmp_aggregate_costs_from_left_direction_.resize(N, 0);

    }
    else if(direction_num == 2)
    {
        tmp_aggregate_costs_from_down_direction_.resize(N, 0); 

    }
    else if(direction_num == 3)
    {
        
        tmp_aggregate_costs_from_up_direction_.resize(N, 0);

    }

}



void PixelPoint::setMinInvDepth(float min_inv_depth)
{

    min_inv_depth_ = min_inv_depth;
    max_depth_ = 1/min_inv_depth;

}

void PixelPoint::setMinDepth(float min_depth)
{
    max_inv_depth_ = 1/min_depth;
    min_depth_ = min_depth;
}

void PixelPoint::setMaxInvDepth(float max_inv_depth)
{
    max_inv_depth_ = max_inv_depth;
    min_depth_ = 1/max_inv_depth;
}

void PixelPoint::setMaxDepth(float max_depth)
{
    min_inv_depth_ = 1/max_depth;
    max_depth_ = max_depth;

}


void PixelPoint::clearMatchInformation()
{
    costs_.clear();
    aggregate_costs_.clear();
    valid_uvs_.clear();
    inv_depths_.clear();
    mini_cost_idx_ = -1;
    mini_aggregate_cost_idx_ = -1;
}



PixelPoint::PixelPoint():
    depth_(0),
    status_(Status::UNINITIALIZED),
    min_inv_depth_(-1),
    max_inv_depth_(-1),
    min_depth_(-1),
    max_depth_(-1),
    num_has_been_aggregated_(0),
    confidence_(0),
    num_correct_consistency_check_(0)
{
    id_ = NPixelPoint_;
    NPixelPoint_++ ;

}


void PixelPoint::setImagePtr(Image* const image)
{
    image_ = image;
    camera_id_ = image->getCameraId();
    pose_id_ = image->getPoseId();
}

bool PixelPoint::DoesExistDepthIntersection(const PixelPoint& ref_pixel_point) const
{   
    bool does_exist_depth_intersection = false;
    if(!(ref_pixel_point.min_depth_ > min_depth_ || ref_pixel_point.max_depth_ < max_depth_))
    {
        does_exist_depth_intersection = true;
    }

    return does_exist_depth_intersection;

}






float PixelPoint::getInterpolatedTmpAggregatedCostByInvDepth(float ref_inv_depth) const
{

    bool is_exist_interpolated_cost = false;
    float interpolated_tmp_aggregated_cost = std::numeric_limits<float>::infinity();

        
    for(int inv_depth_idx = 1; inv_depth_idx < (int)inv_depths_.size(); inv_depth_idx++)
    {
        float lower_inv_depth = inv_depths_[inv_depth_idx-1];
        float inv_depth = inv_depths_[inv_depth_idx];
        if(ref_inv_depth <= inv_depth && ref_inv_depth >= lower_inv_depth)
        {
            interpolated_tmp_aggregated_cost =  tmp_aggregate_costs_[inv_depth_idx-1] + (ref_inv_depth - lower_inv_depth) * 
                (tmp_aggregate_costs_[inv_depth_idx] - tmp_aggregate_costs_[inv_depth_idx-1])/(inv_depth - lower_inv_depth);
            is_exist_interpolated_cost = true;
            break;
        }
    }


    

    return interpolated_tmp_aggregated_cost;
}





void PixelPoint::setUV(int u, int v)
{
    u_ = u;
    v_ = v;
}

void PixelPoint::setIntensity(float intensity)
{
    intensity_ = intensity;
}

void PixelPoint::setGradient(float gradient_u, float gradient_v)
{
    gradient_u_ = gradient_u;
    gradient_v_ = gradient_v;
}

Image::Image(Config* const config)
:ptr_raw_gray_data_(nullptr),
ptr_pixel_point_matrix_(nullptr),
ptr_tar_image_(nullptr),
GT_depth_path_(""),
width_(0),
height_(0),
config_(config)
{   
    id_ = Nimage_;
    Nimage_++ ;
}


Image::~Image(){
    delete[] ptr_raw_gray_data_;
    delete[] ptr_pixel_point_matrix_;

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

std::map<int, Image*>& Image::getMapOfImagesHasBeenMatched()
{
    return map_of_images_has_been_matched_;
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

const std::string Image::getGTDepthPath() const
{
    return GT_depth_path_;
}

void Image::setTimestamp(const double timestamp)
{
    timestamp_ = timestamp;
}

void Image::setTarImagePtr(Image* const ptr_tar_image)
{
    ptr_tar_image_ = ptr_tar_image;
}

void Image::setKRKi(const Eigen::Matrix3f& KRKi)
{
    KRKi_ = KRKi;
}

void Image::setKt(const Eigen::Vector3f& Kt)
{
    Kt_ = Kt;
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

    cv_bgr_data_ = cv::imread(path_, cv::IMREAD_COLOR);

    if (!cv_bgr_data_.empty()) {
        // Successfully read as RGB
        cv::cvtColor(cv_bgr_data_, cv_raw_gray_data, cv::COLOR_BGR2GRAY);
    } else {
        
        std::cout << "Failed to read " << path_ << ". end the program" << std::endl;
    }

    // cv::resize(cv_bgr_data_, cv_mini_bgr_data_, cv::Size(160, 120), 0, 0, cv::INTER_LINEAR);

    if(wOrg_ != cv_raw_gray_data.cols || hOrg_ != cv_raw_gray_data.rows)
    {
        std::cout << "setting of original resolution (" << wOrg_ << "," << hOrg_ <<  ") is conflict with data (" << cv_raw_gray_data.cols << "," <<  cv_raw_gray_data.rows << ")" << std::endl;
        return false;
    }

    whOrg_ = wOrg_ * hOrg_;


    ptr_raw_gray_data_ = new uint8_t[whOrg_]();
    memcpy(ptr_raw_gray_data_, cv_raw_gray_data.data, wOrg_ * hOrg_);

    width_ = config_->cameras_[this->camera_id_].resolution_[0];
    height_ = config_->cameras_[this->camera_id_].resolution_[1];
    wh_ = width_*height_;
    ptr_pixel_point_matrix_ = new PixelPoint[wh_]();
    config_->cameras_[camera_id_].undistort_->undistort(this);



    // Visualizer::showUndistortedGrayImage(this, "undistorted");
    // cv::waitKey(0);

    // ptr_gray_rgb_data_ = new uint8_t[3*wh_]();
    // for(int i=0;i<wh_;i++)
    // {
    //     ptr_gray_rgb_data_[3*i] = ptr_gray_data_[i];
    //     ptr_gray_rgb_data_[3*i+1] = ptr_gray_data_[i];
    //     ptr_gray_rgb_data_[3*i+2] = ptr_gray_data_[i];
    // };


    // ptr_depth_data_ = new float[wh_]();
    // ptr_min_inv_depth_data_ = new float[wh_]();
    // ptr_max_inv_depth_data_ = new float[wh_]();
    // ptr_min_depth_data_ = new float[wh_]();
    // ptr_max_depth_data_ = new float[wh_]();
    
    // cv::Mat depth(height_, width_, CV_32F, (void*)ptr_depth_data_);
    // cv::Mat depth_normalized;
    // cv::normalize(depth, depth_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    // cv::imshow("depth", depth);
    // cv::waitKey(0);

    // just set it infinity far away
    // std::fill(ptr_min_inv_depth_data_, ptr_min_inv_depth_data_ + wh_, 1e-8);
    // std::fill(ptr_max_inv_depth_data_, ptr_max_inv_depth_data_ + wh_, 1.0/config_->min_depth_);

    // ptr_gradient_gray_data_ = new float[2 * wh_]();

    // calGraident();
    calDisablePixelPoint();

    std::cout <<  " we has loaded image with " << id_ << std::endl;


    return true;

}

void Image::calGraident()
{

    // make gradient
    for (int y = 1; y < height_ - 1; ++y)
    {
        for (int x = 1; x < width_ - 1; ++x)
        {
            int cidx = y * width_ + x;

            float Gu = 0.5f * (ptr_pixel_point_matrix_[(y) * width_ + (x + 1)].intensity_ -
                            ptr_pixel_point_matrix_[(y) * width_ + (x - 1)].intensity_);

            float Gv = 0.5f * (ptr_pixel_point_matrix_[(y + 1) * width_ + (x)].intensity_ -
                            ptr_pixel_point_matrix_[(y - 1) * width_ + (x)].intensity_);

            ptr_pixel_point_matrix_[cidx].setGradient(Gu, Gv);
 
        }
    }

}

void Image::calDisablePixelPoint()
{

    float ws_2 = (2*config_->half_window_size_+1)*(2*config_->half_window_size_+1);
    for (int v = 1 + config_->half_window_size_; v < height_ - config_->half_window_size_ - 1; ++v)
    {
        for (int u = 1 + config_->half_window_size_; u < width_ - config_->half_window_size_ - 1; ++u)
        {
            int cidx = v * width_ + u;
            // float sum_ref = 0;
            // for (int y = -config_->half_window_size_; y <= config_->half_window_size_; ++y) {
            //     uint32_t ref_row = (v + y) * width_;

            //     for (int x = -config_->half_window_size_; x <= config_->half_window_size_; ++x) {
            //         sum_ref += ptr_pixel_point_matrix_[ref_row + u + x].intensity_;
            //     }
            // }

            // const float mean_ref = sum_ref / (float)ws_2;

            // float self_cost_left = ZSAD(ptr_pixel_point_matrix_, u, v, mean_ref, ptr_pixel_point_matrix_, u-1, v, width_, height_, config_->half_window_size_)/(ws_2);
            // float self_cost_right = ZSAD(ptr_pixel_point_matrix_, u, v, mean_ref, ptr_pixel_point_matrix_, u+1, v, width_, height_, config_->half_window_size_)/(ws_2);
            // float self_cost_up = ZSAD(ptr_pixel_point_matrix_, u, v, mean_ref, ptr_pixel_point_matrix_, u, v+1, width_, height_, config_->half_window_size_)/(ws_2);
            // float self_cost_down = ZSAD(ptr_pixel_point_matrix_, u, v, mean_ref, ptr_pixel_point_matrix_, u, v-1, width_, height_, config_->half_window_size_)/(ws_2);
            
            // float self_cost_left_up = ZSAD(ptr_pixel_point_matrix_, u, v, mean_ref, ptr_pixel_point_matrix_, u-1, v+1, width_, height_, config_->half_window_size_)/(ws_2);
            // float self_cost_left_down = ZSAD(ptr_pixel_point_matrix_, u, v, mean_ref, ptr_pixel_point_matrix_, u-1, v-1, width_, height_, config_->half_window_size_)/(ws_2);
            // float self_cost_right_up = ZSAD(ptr_pixel_point_matrix_, u, v, mean_ref, ptr_pixel_point_matrix_, u+1, v+1, width_, height_, config_->half_window_size_)/(ws_2);
            // float self_cost_right_down = ZSAD(ptr_pixel_point_matrix_, u, v, mean_ref, ptr_pixel_point_matrix_, u+1, v-1, width_, height_, config_->half_window_size_)/(ws_2);

            float self_cost_left = SAD(ptr_pixel_point_matrix_, u, v, width_, ptr_pixel_point_matrix_, u-1, v, width_, config_->half_window_size_)/(ws_2);
            float self_cost_right = SAD(ptr_pixel_point_matrix_, u, v, width_, ptr_pixel_point_matrix_, u+1, v, width_, config_->half_window_size_)/(ws_2);
            float self_cost_up = SAD(ptr_pixel_point_matrix_, u, v, width_, ptr_pixel_point_matrix_, u, v+1, width_, config_->half_window_size_)/(ws_2);
            float self_cost_down = SAD(ptr_pixel_point_matrix_, u, v, width_, ptr_pixel_point_matrix_, u, v-1, width_, config_->half_window_size_)/(ws_2);
            
            float self_cost_left_up = SAD(ptr_pixel_point_matrix_, u, v, width_, ptr_pixel_point_matrix_, u-1, v+1, width_, config_->half_window_size_)/(ws_2);
            float self_cost_left_down = SAD(ptr_pixel_point_matrix_, u, v, width_, ptr_pixel_point_matrix_, u-1, v-1, width_, config_->half_window_size_)/(ws_2);
            float self_cost_right_up = SAD(ptr_pixel_point_matrix_, u, v, width_, ptr_pixel_point_matrix_, u+1, v+1, width_, config_->half_window_size_)/(ws_2);
            float self_cost_right_down = SAD(ptr_pixel_point_matrix_, u, v, width_, ptr_pixel_point_matrix_, u+1, v-1, width_, config_->half_window_size_)/(ws_2);


            // float self_cost_left = ASW(ptr_pixel_point_matrix_, u, v, width_, ptr_pixel_point_matrix_, u-1, v, width_, config_->half_window_size_);
            // float self_cost_right = ASW(ptr_pixel_point_matrix_, u, v, width_, ptr_pixel_point_matrix_, u+1, v, width_, config_->half_window_size_);
            // float self_cost_up = ASW(ptr_pixel_point_matrix_, u, v, width_, ptr_pixel_point_matrix_, u, v+1, width_, config_->half_window_size_);
            // float self_cost_down = ASW(ptr_pixel_point_matrix_, u, v, width_, ptr_pixel_point_matrix_, u, v-1, width_, config_->half_window_size_);
            
            // float self_cost_left_up = ASW(ptr_pixel_point_matrix_, u, v, width_, ptr_pixel_point_matrix_, u-1, v+1, width_, config_->half_window_size_);
            // float self_cost_left_down = ASW(ptr_pixel_point_matrix_, u, v, width_, ptr_pixel_point_matrix_, u-1, v-1, width_, config_->half_window_size_);
            // float self_cost_right_up = ASW(ptr_pixel_point_matrix_, u, v, width_, ptr_pixel_point_matrix_, u+1, v+1, width_, config_->half_window_size_);
            // float self_cost_right_down = ASW(ptr_pixel_point_matrix_, u, v, width_, ptr_pixel_point_matrix_, u+1, v-1, width_, config_->half_window_size_);

            

            float self_cost = std::max(std::max(std::max(std::max(std::max(std::max(std::max(self_cost_left, self_cost_right), self_cost_up), self_cost_down), self_cost_left_up), self_cost_left_down), self_cost_right_up), self_cost_right_down);
            
            if(self_cost <= ACCEPTABLE_COST_DIFF)
            {
                ptr_pixel_point_matrix_[cidx].status_ = PixelPoint::Status::DISABLE;
            }
 
        }
    }
}


uint32_t Image::getWidthOrg() const
{
    return wOrg_;
}

uint32_t Image::getHeightOrg() const
{
    return hOrg_;

}

uint32_t Image::getWidth() const
{
    return width_;
}

uint32_t Image::getHeight() const
{
    return height_;
}


void Image::setWidthOrg(uint32_t widthOrg)
{
    wOrg_ = widthOrg;
}

void Image::setHeightOrg(uint32_t heightOrg)
{
    hOrg_ = heightOrg;
}


uint8_t* Image::getRawGrayDataPtr() const
{
    return ptr_raw_gray_data_;
}


PixelPoint* Image::getPixelPointMatrixPtr() const
{
    return ptr_pixel_point_matrix_;
}



const cv::Mat& Image::getBGRData() const
{
    return cv_bgr_data_;
}

const Eigen::Matrix3f& Image::getKRKi() const
{
    return KRKi_;
}

const Eigen::Vector3f& Image::getKt() const
{
    return Kt_;
}

// const cv::Mat& Image::getMiniBGRData() const
// {
//     return cv_mini_bgr_data_;
// }

const cv::Mat& Image::getGTDepthData() const
{
    return cv_gt_depth_data_;
}




cv::Mat Image::getCVDepth() const
{

    cv::Mat depth(height_, width_, CV_32F);

    for (int v = 0; v < height_; ++v) {
        float* row = depth.ptr<float>(v);
        for (int u = 0; u < width_; ++u) {
            const int i = v * width_ + u;
            const PixelPoint& p = ptr_pixel_point_matrix_[i];
            row[u] = p.depth_;                
        }
    }

    return depth;
}


bool Image::isInImage(float u, float v, int border) const
{
    if(ptr_pixel_point_matrix_!= nullptr)
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
