#pragma once

#include <Eigen/Dense>

#include <glk/mesh.hpp>
#include <glk/texture_opencv.hpp>
#include <guik/viewer/shader_setting.hpp>
#include <glk/primitives/primitives.hpp>
#include <glk/thin_lines.hpp>
#include <glk/lines.hpp>
#include <glk/drawable.hpp>
#include <glk/texture.hpp>
#include <glk/pointcloud_buffer.hpp>

#include <guik/viewer/light_viewer.hpp>
#include <guik/viewer/light_viewer_context.hpp>

#include <implot.h>


#include <unordered_map> 

#include "visualizer.h"

namespace MVS{



class CameraModel
{

    public:


    CameraModel(const float size = 1.0f, const Eigen::Vector4f& color = {0.0f, 1.0f, 0.0f, 1.0f});
    ~CameraModel();

    void draw(guik::LightViewer* viewer, std::string name) const;

    // void setTransparent(const bool isTransparent);
    void setColor(const Eigen::Vector4f& color);
    void setTransform(const Eigen::Matrix4f& transform);

    protected:

    int id_ = 0;

    std::shared_ptr<glk::Mesh> frontside_ = nullptr;
    std::shared_ptr<guik::ShaderSetting> frontside_shader_ = nullptr;

    std::shared_ptr<glk::ThinLines> backbone_ = nullptr;
    std::shared_ptr<guik::ShaderSetting> backbone_shader_ = nullptr;


    std::shared_ptr<glk::Lines> coord_x_axis_ = nullptr;
    std::shared_ptr<guik::ShaderSetting> coord_x_axis_shader_ = nullptr;
    std::shared_ptr<glk::Lines> coord_y_axis_ = nullptr;
    std::shared_ptr<guik::ShaderSetting> coord_y_axis_shader_ = nullptr;
    std::shared_ptr<glk::Lines> coord_z_axis_ = nullptr;
    std::shared_ptr<guik::ShaderSetting> coord_z_axis_shader_ = nullptr;
 


};





class IridescenceVisualizer : public Visualizer
{

    public:

    IridescenceVisualizer(Config* const config);
    ~IridescenceVisualizer() = default;


    void showRefImageReconstruction(Image* const ref_image, Image* const tar_image) override;
    void DebugPlot();

    protected:


    guik::LightViewer* viewer_;
    std::shared_ptr<guik::LightViewerContext> depth_uncertainty_viewer_;



    Eigen::Matrix4f Tcl_;

    bool isProcessNextTarImage_ = false;

    std::shared_ptr<glk::Mesh> ref_image_canvas_;

    Eigen::Vector3f picked_uncertainty_point_position_;
    PixelPoint* ptr_picked_ref_ptr_pixel_point_;

    Image* ref_image_;
    Image* tar_image_;  



};



}  // namespace MVS

