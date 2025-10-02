#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


#include "configs.h"
#include "data.h"
#include "multi_view_stereo.h"


int main(int argv, char ** argc)
{

    std::string config_path = argc[1];
    cv::FileStorage* fs = new cv::FileStorage(config_path, cv::FileStorage::READ);
    if (!fs->isOpened()) {
        std::cout << "Failed to open YAML file for path " <<  config_path << "." << std::endl;
        return -1;
    }

    MVS::Config* config = new MVS::Config(fs);

    MVS::Dataset* dataset = new MVS::Dataset(config);

    MVS::MultiViewStereo multi_view_stereo(config);
    multi_view_stereo.setDataset(dataset);
    multi_view_stereo.run();


    return 0;

}