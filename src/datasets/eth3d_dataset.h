#pragma once
#include "dataset.h"



namespace MVS
{




class ETH3DDataset : public Dataset
{


    public:

    ETH3DDataset(Config* const config);
    
    void readTrajectory() override;
    // bool loadGTDepth(Image* const image) override;


    private:


};


}