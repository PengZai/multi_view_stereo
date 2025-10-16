#pragma once
#include "dataset.h"

namespace MVS
{


class FastLivo2Dataset : public Dataset
{


    public:

    FastLivo2Dataset(Config* const config);
    
    void readTrajectory() override;


};

    
    
} // namespace MVS



