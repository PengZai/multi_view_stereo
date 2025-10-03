#include "visualizer.h"




namespace MVS
{

Visualizer::Visualizer(Config* const config)
{
    config_ = config;
}

Visualizer::~Visualizer()
{

}

void Visualizer::showDepth(const Image* const image)
{
    const float* const ptr_depth = image->getDepthPtr();
    int width = image->getWidth();
    int height = image->getHeight();

    cv::Mat depth(height, width, CV_32F, (void*)ptr_depth);
    cv::Mat depth_normalized;
    cv::normalize(depth, depth_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat depth_colormap;
    cv::applyColorMap(depth_normalized, depth_colormap, cv::COLORMAP_JET);
    cv::imshow("Depth Map", depth_colormap);
    cv::imwrite(config_->save_figure_path_+"/Depth_map.png", depth_colormap);


}

} // namespace MVS



