#pragma once
#include <opencv2/opencv.hpp>
#include "data.h"
#include "configs.h"

namespace MVS
{


class Visualizer
{   
    public:
    Visualizer(Config* const config);
    ~Visualizer();

    void showDepth(const Image* const image);

    private:
    Config* config_;
};
    
} // namespace MVS

