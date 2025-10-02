#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <string>
#include <filesystem>

#include "configs.h"

namespace MVS
{



class Image
{
    public:

        static int Nimage_;

        Image();
        const Eigen::Vector3f& getTranslation() const;
        void setTranslation(const Eigen::Vector3f &translation);
        const Eigen::Quaternionf& getQuaternion() const;
        void setQuaternion(const Eigen::Quaternionf &quaternion);
        const Eigen::Matrix3f getRotationMatrix() const;
        const Eigen::Matrix4f getTransformationMatrix() const;
        void setName(const std::string &name);
        void setPath(const std::string &path);
        void setCameraId(const int camera_id);
        int getCameraId();
        int getId();

        void setTimestamp(const double timestamp);
        void loadData();
        cv::Mat getGrayData() const;
        cv::Mat getRGBData() const;
        bool isInImage(float u, float v, int border=0) const;


    private:
        
        std::string name_;
        std::string path_;
        int id_;
        int camera_id_;
        double timestamp_;
        cv::Mat rgb_data_;
        cv::Mat gray_data_;
        cv::Mat depth_;
        cv::Mat deep_learning_depth_;

        Eigen::Vector3f t_;
        Eigen::Quaternionf q_;
    

    

};


class Dataset
{
    public:
    Dataset(Config* const config);
    ~Dataset();

    std::vector<Image*> images_;
    std::vector<std::string> image_dir_paths_;

    private:
    void readTrajectory(const Camera *cameras, const std::string& path, int maximum_traj);


};



} // namespace MVS
