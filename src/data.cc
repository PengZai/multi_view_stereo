#include "data.h"


namespace MVS
{

int Image::Nimage_ = 0;

Dataset::Dataset(Config* const config)
{


    readTrajectory(config->cameras_, config->trajectory_path_, config->maximum_traj_);
    
}

Dataset::~Dataset(){
    for (auto* p : images_) delete p;
    images_.clear();
}

void Dataset::readTrajectory(const Camera *cameras, const std::string& path, int maximum_traj)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << path << std::endl;
    }

    std::string line;
    int Nline = 0;
    int Ntraj = 0;
    while (std::getline(file, line)) 
    {
        // skip comments
        if (line.empty() || line[0] == '#'){
            Nline++;
            continue;
        }
        std::istringstream iss(line);
        Image *image = new Image();
        int cam_id;
        std::string str_timestamp;
        float x, y, z, qx, qy, qz, qw;
        if (!(iss >> cam_id >> str_timestamp >> x >> y >> z >> qx >> qy >> qz >> qw)) {
            std::cout << " skip malformed lines at " << Nline << std::endl;
            Nline++;
            continue; 
        }
        double timestamp = std::stod(str_timestamp);
        str_timestamp.erase(std::remove(str_timestamp.begin(), str_timestamp.end(), '.'), str_timestamp.end());

        // std::string str_timestamp = std::to_string(timestamp);

        // T_world_camera
        image->setTranslation(Eigen::Vector3f(x,y,z));
        image->setQuaternion(Eigen::Quaternionf(qw,qx,qy,qz));
        image->setName(str_timestamp);
        image->setPath(cameras[cam_id].dir_path_ + "/" + str_timestamp+".png");
        image->setTimestamp(timestamp);
        image->setCameraId(cam_id);
        image->loadData();

        images_.push_back(image);

        Nline++;
        Ntraj++;
        if(maximum_traj > 0 && Ntraj >= maximum_traj){
            break;
        }
        

    }

}

Image::Image()
{
    id_ = Nimage_;
    Nimage_++ ;
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

void Image::setPath(const std::string &path)
{
    path_ = path;
}

int Image::getId()
{
    return id_;
}

void Image::setCameraId(const int camera_id)
{
    camera_id_ = camera_id;
}

int Image::getCameraId()
{
    return camera_id_;
}

void Image::setTimestamp(const double timestamp)
{
    timestamp_ = timestamp;
}

void Image::loadData()
{
    if(path_.empty())
    {
        std::cout <<  "path of image id :" << id_ << " is empty " << std::endl;
    }

    rgb_data_ = cv::imread(path_, cv::IMREAD_COLOR);
    gray_data_ = cv::imread(path_, cv::IMREAD_GRAYSCALE);
 
}


cv::Mat Image::getGrayData() const
{
    return gray_data_;
}

cv::Mat Image::getRGBData() const
{
    return rgb_data_;
}


bool Image::isInImage(float u, float v, int border) const
{
    if(gray_data_.empty()==false)
    {
        // int int_u = round(u);
        // int int_v = round(v);
        // if(int_u>=border && int_v>= border && int_u < gray_data_.cols - border && int_v < gray_data_.rows - border)
        // {
        //     return true;
        // }
        if(u >=border && v >= border && u < gray_data_.cols - border - 1 && v < gray_data_.rows - border - 1)
        {
            return true;
        }
    }

    return false;
}

}
