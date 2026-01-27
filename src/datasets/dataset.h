#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>   
#include <sstream>   
#include <stdexcept> 
#include "../configs.h"

namespace MVS
{

class Config;
class Camera;
class Image;


class OutputData{
    public:

    OutputData(){};
    ~OutputData()= default;

    float depth_;
    float minimum_peak_aggregate_cost_;
    float uncertainty_;

};




class DebugInfo
{

    public:

    DebugInfo(){};
    ~DebugInfo() = default;

    void clear();

    // int tar_pose_id_;
    // int tar_camera_id_;

    void setZerosTmpAggregateCostsFromCertainDirection(const int N, int direction_num);
    void appendTmpAggregateCostsFromCertainDirection(float tmp_aggregate_cost, int direction_num);

    std::vector<float> steps_;
    std::vector<float> tmp_aggregate_costs_from_left_direction_;
    std::vector<float> tmp_aggregate_costs_from_right_direction_;
    std::vector<float> tmp_aggregate_costs_from_up_direction_;
    std::vector<float> tmp_aggregate_costs_from_down_direction_;

    
    // int best_step_idx_;
    int mini_aggregate_cost_idx_;
    int mini_aggregate_cost_idx_minus_;
    int mini_aggregate_cost_idx_plus_;

    int manual_step_idx_ = 0;

    Eigen::Vector2f uv_min_;
    Eigen::Vector2f uv_max_;
    // Eigen::Vector2f uv_best_match_;
    Eigen::Vector2f uv_possible_match_;
    Eigen::Vector2f unit_epipolar_vector_;
    float epipolar_length_;

    float init_min_depth_;
    float init_max_depth_;
    float init_depth_;

    float updated_min_depth_;
    float updated_max_depth_;
    float updated_depth_;

};


class PixelPoint
{
    public:

    enum Status {
        UNINITIALIZED = 0,                // a point have done nothing 
        DISABLE,                          // disable point
        NORMAL,                           // normal situation
        INVALID_COST_OBSERVATION,         // cost match is invalid
        INVALID_AGGREGATION_COST_OBSERVATION,  // aggregation cost is invalid
        INVALID_NON_UNIQUENESSS,          // there are many ambiguous
        READ_TO_SHOW                      // low uncertainty depth and ready to show
    };

    static inline std::string StatusToString(Status v)
    {
        switch (v)
        {
            case UNINITIALIZED:   return "UNINITIALIZED";
            case DISABLE:   return "DISABLE";
            case NORMAL:   return "NORMAL";
            case INVALID_COST_OBSERVATION: return "INVALID_COST_OBSERVATION";
            case INVALID_NON_UNIQUENESSS:   return "INVALID_NON_UNIQUENESSS";
            case READ_TO_SHOW: return "READ_TO_SHOW";
            default:      return "[Unknown OS_type]";
        }
    }


    static int NPixelPoint_;

    PixelPoint();

    void setImagePtr(Image* const image);

    void setUV(int u, int v);

    void setIntensity(float intensity);

    void setGradient(float gradient_u, float gradient_v);

    void setMinInvDepth(float min_inv_depth);
    void setMinDepth(float min_depth);

    void setMaxInvDepth(float max_inv_depth);
    void setMaxDepth(float max_depth);

    void clearMatchInformation();

    float getInterpolatedTmpAggregatedCostByInvDepth(float ref_inv_depth) const;

    bool DoesExistDepthIntersection(const PixelPoint& ref_pixel_point) const;

    ~PixelPoint() = default;

    Image* image_;
    int id_;
    int camera_id_;
    int pose_id_;
    
    Status status_;


    int u_;
    int v_;

    float intensity_;
    float gradient_u_;
    float gradient_v_;
    float depth_;
    
    Eigen::Vector2f unit_epipolar_vector_;

    float min_inv_depth_;
    float min_depth_;

    float max_inv_depth_;
    float max_depth_;
    float confidence_; // 
    int num_correct_consistency_check_;

    OutputData output_data_;


    std::vector<float> costs_;
    std::vector<float> tmp_aggregate_costs_;
    std::vector<float> aggregate_costs_;
    std::vector<Eigen::Vector2f> valid_uvs_;
    std::vector<float> inv_depths_;
    int mini_cost_idx_;
    int mini_aggregate_cost_idx_;
    int num_has_been_aggregated_;
    DebugInfo debug_info_;


};

class Image
{
    public:

        static int Nimage_;

        Image(Config* const config);
        ~Image();

        const Eigen::Vector3f& getTranslation() const;
        void setTranslation(const Eigen::Vector3f &translation);
        const Eigen::Quaternionf& getQuaternion() const;
        void setQuaternion(const Eigen::Quaternionf &quaternion);
        const Eigen::Matrix3f getRotationMatrix() const;
        const Eigen::Matrix4f getTransformationMatrix() const;
        std::map<int, Image*>& getMapOfImagesHasBeenMatched();

        void setName(const std::string &name);
        void setPath(const std::string &path);
        void setGTDepthPath(const std::string &path);
        void setGTDepth(const cv::Mat& cv_gt_depth_data);
        void setCameraId(const int camera_id);
        void setPoseId(const int camera_id);
        void setTimestamp(const double timestamp);
        void setTarImagePtr(Image* const ptr_tar_image);
        void setKRKi(const Eigen::Matrix3f& KRKi);
        void setKt(const Eigen::Vector3f& Kt);
        void setWidthOrg(uint32_t widthOrg);
        void setHeightOrg(uint32_t heightOrg);
    

        int getCameraId() const;
        int getPoseId() const;
        int getId() const;
        const std::string getGTDepthPath() const;
        bool loadData();
        void calGraident();
        void calDisablePixelPoint();

        std::string getImageName() const;
        uint32_t getWidthOrg() const;
        uint32_t getHeightOrg() const;
        uint32_t getWidth() const;
        uint32_t getHeight() const;
        void loadDepthFromMinMaxInvDepth() const;
        uint8_t* getRawGrayDataPtr() const;
        PixelPoint* getPixelPointMatrixPtr() const;
        const cv::Mat& getBGRData() const;
        // const cv::Mat& getMiniBGRData() const;
        const cv::Mat& getGTDepthData() const;
        bool isInImage(float u, float v, int border=0) const;
        cv::Mat getCVDepth() const;
        const Eigen::Matrix3f& getKRKi() const;
        const Eigen::Vector3f& getKt() const;

        Config* config_;


    protected:
        
        std::string name_;
        std::string path_;
        int id_;
        int camera_id_;
        int pose_id_;
        double timestamp_;
        int wOrg_;
        int hOrg_;
        int whOrg_;
        int width_;
        int height_;
        int wh_;
        cv::Mat cv_bgr_data_;

        uint8_t* ptr_raw_gray_data_;
        PixelPoint* ptr_pixel_point_matrix_;

        std::map<int, Image*> map_of_images_has_been_matched_;


        Eigen::Vector3f t_;
        Eigen::Quaternionf q_;

        std::string GT_depth_path_;
        cv::Mat cv_gt_depth_data_;

        // tar image characteristic
        Image* ptr_tar_image_;
        Eigen::Matrix3f KRKi_; // KRKi = K_tar * R_tar_ref * K_ref.inverse();
        Eigen::Vector3f Kt_; // K_tar * t_tar_ref;
        
        float ACCEPTABLE_COST_DIFF = 2;


};


class Dataset
{
    public:
    Dataset(Config* const config);
    virtual ~Dataset();
    std::vector<Image*>& getImages();
    virtual void readTrajectory() = 0;
    virtual bool loadGTDepth(Image* const image);


    protected:

    Config* config_;
    std::vector<Image*> images_;
    Camera *cameras_;
    // std::string trajectory_path_;
    bool is_all_cameras_in_same_traj_file_;
    int maximum_traj_;
    int minimum_traj_;
};


Dataset* getDataset(Config* const config);


} // namespace MVS
