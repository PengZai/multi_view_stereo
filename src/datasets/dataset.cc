#include "dataset.h"
#include "botanic_gardent_dataset.h"
#include "fast_livo2_dataset.h"
#include "remode_dataset.h"
#include "tartan_air_dataset.h"
#include "kitti_dataset.h"
#include "tanks_temples_dataset.h"
#include "eth3d_dataset.h"
// #include "virtual_kitti_dataset.h"
#include "../visualizer.h"


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

    uv_min_ = Eigen::Vector2f::Zero();
    uv_max_ = Eigen::Vector2f::Zero();
    // uv_best_match_ = Eigen::Vector2f::Zero();
    unit_epipolar_vector_ = Eigen::Vector2f::Zero();
    epipolar_length_ = -1;

    init_min_depth_ = -1;
    init_max_depth_ = -1;
    init_depth_ = -1;


}


EpipolarSegment::EpipolarSegment(float min_inv_depth, float max_inv_depth):
min_inv_depth_(min_inv_depth),
max_inv_depth_(max_inv_depth),
min_depth_(1/max_inv_depth),
max_depth_(1/min_inv_depth)
{


}

EpipolarSegment::EpipolarSegment():
EpipolarSegment(-1, -1)
{

}


void EpipolarSegment::setMinInvDepth(float min_inv_depth)
{

    min_inv_depth_ = min_inv_depth;
    max_depth_ = 1/min_inv_depth;

}

void EpipolarSegment::setMinDepth(float min_depth)
{
    max_inv_depth_ = 1/min_depth;
    min_depth_ = min_depth;
}

void EpipolarSegment::setMaxInvDepth(float max_inv_depth)
{
    max_inv_depth_ = max_inv_depth;
    min_depth_ = 1/max_inv_depth;
}

void EpipolarSegment::setMaxDepth(float max_depth)
{
    min_inv_depth_ = 1/max_depth;
    max_depth_ = max_depth;

}


void EpipolarSegment::clear()
{
    costs_.clear();
    aggregate_costs_.clear();
    valid_uvs_.clear();
    inv_depths_.clear();
}



PixelPoint::PixelPoint():
    depth_(0),
    status_(Status::UNINITIALIZED)
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

bool PixelPoint::DoesExistDepthIntersection(const PixelPoint& ref_pixel_point, int minimum_cost_epipolar_segment_idx) const
{   
    bool does_exist_depth_intersection = false;
    const EpipolarSegment& epipolar_segment_with_minimum_cost_on_ref_pixel_point = ref_pixel_point.epipolar_segment_vec_[minimum_cost_epipolar_segment_idx];

    for(int epipolar_segment_idx=0; epipolar_segment_idx<(int)epipolar_segment_vec_.size(); epipolar_segment_idx++)
    {
        const EpipolarSegment& epipolar_segment = epipolar_segment_vec_[epipolar_segment_idx];
        if(!(epipolar_segment_with_minimum_cost_on_ref_pixel_point.min_inv_depth_ > epipolar_segment.max_inv_depth_ || epipolar_segment_with_minimum_cost_on_ref_pixel_point.max_inv_depth_ < epipolar_segment.min_inv_depth_))
        {
            does_exist_depth_intersection = true;
            break;
        }
    }
    
    return does_exist_depth_intersection;

}


void PixelPoint::getMininumIdxTmpAggregatedCost(int &minimum_cost_epipolar_segment_idx, int &minimum_cost_idx) const
{

    float mini_cost = std::numeric_limits<float>::infinity();
    for(int epipolar_segment_idx=0; epipolar_segment_idx < (int)epipolar_segment_vec_.size(); epipolar_segment_idx++)
    {

        const EpipolarSegment& epipolar_segment = epipolar_segment_vec_[epipolar_segment_idx];
        for(int cost_idx = 0; cost_idx < (int)epipolar_segment.tmp_aggregate_costs_.size(); cost_idx++ )
        {
            if(epipolar_segment.tmp_aggregate_costs_[cost_idx] < mini_cost)
            {
                mini_cost = epipolar_segment.tmp_aggregate_costs_[cost_idx];
                minimum_cost_epipolar_segment_idx = epipolar_segment_idx;
                minimum_cost_idx = cost_idx;
            } 
        }
    }


}

void PixelPoint::getGlobalMininumPeakIdx(int &global_minimum_peak_epipolar_segment_idx, int &global_minimum_peak_idx) const
{

    float mini_cost = std::numeric_limits<float>::infinity();
    global_minimum_peak_epipolar_segment_idx = -1;
    global_minimum_peak_idx = -1;

    for(int epipolar_segment_idx=0; epipolar_segment_idx < (int)epipolar_segment_vec_.size(); epipolar_segment_idx++)
    {
        const EpipolarSegment& epipolar_segment = epipolar_segment_vec_[epipolar_segment_idx];
        if(epipolar_segment.local_minimum_peaks_.size() == 0){
            continue;
        }

        const MinimumPeak& local_minimum_peak = epipolar_segment.local_minimum_peaks_[0];
        if(mini_cost > epipolar_segment.aggregate_costs_[local_minimum_peak.born_idx_])
        {
            mini_cost = epipolar_segment.aggregate_costs_[local_minimum_peak.born_idx_];
            global_minimum_peak_epipolar_segment_idx = epipolar_segment_idx;
            global_minimum_peak_idx = 0;
        }

    }



}



float PixelPoint::getInterpolatedTmpAggregatedCostByInvDepth(float ref_inv_depth) const
{

    bool is_exist_interpolated_cost = false;
    float interpolated_tmp_aggregated_cost = std::numeric_limits<float>::infinity();
    for(int epipolar_segment_idx = 0; epipolar_segment_idx < (int)epipolar_segment_vec_.size(); epipolar_segment_idx++)
    {
        
        const EpipolarSegment& epipolar_segment = epipolar_segment_vec_[epipolar_segment_idx];

        for(int inv_depth_idx = 1; inv_depth_idx < (int)epipolar_segment.inv_depths_.size(); inv_depth_idx++)
        {
            float lower_inv_depth = epipolar_segment.inv_depths_[inv_depth_idx-1];
            float inv_depth = epipolar_segment.inv_depths_[inv_depth_idx];
            if(ref_inv_depth <= inv_depth && ref_inv_depth >= lower_inv_depth)
            {
                interpolated_tmp_aggregated_cost =  epipolar_segment.tmp_aggregate_costs_[inv_depth_idx-1] + (ref_inv_depth - lower_inv_depth) * (epipolar_segment.tmp_aggregate_costs_[inv_depth_idx] - epipolar_segment.tmp_aggregate_costs_[inv_depth_idx-1])/(inv_depth - lower_inv_depth);
                is_exist_interpolated_cost = true;
                break;
            }
        }

        if(is_exist_interpolated_cost == true)
        {
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

    wOrg_ = cv_raw_gray_data.cols;
    hOrg_ = cv_raw_gray_data.rows;
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

    calGraident();

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
            PixelPoint& pixel_point = ptr_pixel_point_matrix_[current_coord];
            float min_depth = config_->min_depth_;
            float max_depth = config_->max_depth_;

            if(pixel_point.epipolar_segment_vec_.size() >= 1)
            {
                min_depth = 1.0/pixel_point.epipolar_segment_vec_[0].max_inv_depth_;
                max_depth = 1.0/pixel_point.epipolar_segment_vec_[0].min_inv_depth_;
            }

            float diff = abs(max_depth-min_depth);
            float depth = (min_depth+max_depth)/2.0;
            if(depth >= config_->min_depth_ && depth <= config_->max_depth_){
                pixel_point.depth_ = depth;
            }

        }
    }

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
