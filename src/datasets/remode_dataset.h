#pragma once

#include "dataset.h"


namespace MVS
{

    
class RemodeDataset : public MVS::Dataset
{
    public:
    RemodeDataset(Config* const config);

    void readTrajectory() override;
    bool loadGTDepth(const std::string& path, cv::Mat &cv_gt_depth_data, const int width, const int height);

};


    
    
} // namespace MVS

