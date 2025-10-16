#pragma once
#include "dataset.h"

namespace MVS
{


class BotanicGardenDataset : public Dataset
{


    public:

    BotanicGardenDataset(Config* const config);
    
    void readTrajectory() override;


};

    
    
} // namespace MVS



