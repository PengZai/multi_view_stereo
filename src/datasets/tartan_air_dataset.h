#pragma once
#include "dataset.h"



namespace MVS
{


class TartanAirDataset : public Dataset
{


    public:

    TartanAirDataset(Config* const config);
    
    void readTrajectory() override;
    // bool loadGTDepth(Image* const image) override;


};


}