#pragma once
#include "dataset.h"

namespace MVS
{


class KittiDataset : public Dataset
{


    public:

    KittiDataset(Config* const config);
    
    void readTrajectory() override;


};

    
    
} // namespace MVS



