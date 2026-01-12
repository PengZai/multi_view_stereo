#pragma once
#include "dataset.h"



namespace MVS
{




class TanksTemplesDataset : public Dataset
{


    public:

    TanksTemplesDataset(Config* const config);
    
    void readTrajectory() override;
    // bool loadGTDepth(Image* const image) override;


    private:


};


}