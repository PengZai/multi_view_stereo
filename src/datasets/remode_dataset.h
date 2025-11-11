#pragma once

#include "dataset.h"


namespace MVS
{

    
class RemodeDataset : public MVS::Dataset
{
    public:
    RemodeDataset(Config* const config);

    void readTrajectory() override;
    bool loadGTDepth(Image* const image) override;

};


    
    
} // namespace MVS

