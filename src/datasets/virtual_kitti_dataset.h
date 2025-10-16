#pragma once
#include "dataset.h"

namespace MVS
{


class VirtualKittiDataset : public Dataset
{


    public:

    VirtualKittiDataset(Config* const config);
    
    void readTrajectory() override;


};

    
    
} // namespace MVS



