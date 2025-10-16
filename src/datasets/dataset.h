#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <string>
#include <filesystem>

#include "../configs.h"


namespace MVS
{

class Config;
class Camera;
class BotanicGardenDataset;
class FastLivo2Dataset;
class RemodeDataset;
class TartanAirDataset;
class KittiDataset;
class VirtualKittiDataset;

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
        void setName(const std::string &name);
        void setPath(const std::string &path);
        void setGTDepthPath(const std::string &path);
        void setGTDepth(const cv::Mat& cv_gt_depth_data);
        void setCameraId(const int camera_id);
        void setPoseId(const int camera_id);
        int getCameraId() const;
        int getPoseId() const;
        int getId() const;
        void setTimestamp(const double timestamp);
        bool loadData();
        void makeGraident();

        std::string getImageName() const;
        uint32_t getWidth() const;
        uint32_t getHeight() const;
        void loadDepthFromMinMaxInvDepth() const;
        uint8_t* getRawGrayDataPtr() const;
        float* getGrayDataPtr() const;
        float* getMinInvDepthDataPtr() const;
        float* getMaxInvDepthDataPtr() const;
        float* getGradientGrayDataPtr() const;
        float* getDepthPtr() const;
        const cv::Mat& getRGBData() const;
        const cv::Mat& getGTDepthData() const;
        bool isInImage(float u, float v, int border=0) const;


    protected:
        
        Config* config_;
        std::string name_;
        std::string path_;
        int id_;
        int camera_id_;
        int pose_id_;
        double timestamp_;
        int width_;
        int height_;
        cv::Mat cv_rgb_data_;
        uint8_t* ptr_raw_gray_data_;
        float* ptr_gray_data_;
        float* ptr_gradient_gray_data_;

        float* ptr_depth_data_;
        float* ptr_min_inv_depth_data_;
        float* ptr_max_inv_depth_data_;

        Eigen::Vector3f t_;
        Eigen::Quaternionf q_;

        std::string GT_depth_path_;
        cv::Mat cv_gt_depth_data_;
    

    

};


class Dataset
{
    public:
    Dataset(Config* const config);
    virtual ~Dataset();
    std::vector<Image*>& getImages();
    virtual void readTrajectory() = 0;

    protected:

    Config* config_;
    std::vector<Image*> images_;
    Camera *cameras_;
    // std::string trajectory_path_;
    bool is_all_cameras_in_same_traj_file_;
    int maximum_traj_;

};


Dataset* getDataset(Config* const config);


} // namespace MVS
