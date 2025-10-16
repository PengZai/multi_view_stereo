#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


#include "configs.h"
#include "datasets/dataset.h"
#include "multi_view_stereo.h"


int main(int argv, char ** argc)
{

    const std::string config_path = argc[1]; 

    MVS::Config* config = new MVS::Config(config_path);

    MVS::Dataset* dataset = getDataset(config);

    MVS::MultiViewStereo multi_view_stereo(config);
    multi_view_stereo.setDataset(dataset);
    multi_view_stereo.run();


    // delete dataset;

    return 0;

}