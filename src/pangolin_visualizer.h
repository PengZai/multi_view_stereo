#pragma once

#include <pangolin/pangolin.h>
#include <pangolin/gl/glfont.h>   // make sure this is included
#include <pangolin/gl/gltext.h>
#include <pangolin/display/default_font.h>

#include "visualizer.h"


namespace MVS
{


class PangolinVisualizer : public Visualizer
{   
    public:
    PangolinVisualizer(Config* const config);
    ~PangolinVisualizer();
   
    pangolin::View& getPangolinViewer();
    pangolin::OpenGlRenderState& getPangolinRenderState();
    void drawFrame(const Eigen::Matrix4f &T_w_c, const Eigen::Vector4i &bgra, bool drawAxis, const std::string &text);
    void drawPoint(const Eigen::Vector3f &pt3f, const Eigen::Vector3i &bgr);
    void drawPoint(const Eigen::Vector3f &pt3f, float gray);
    void showInterface() override;


    protected:
    pangolin::Var<bool>* isProcessNextTarImage_;
    pangolin::Var<bool>* isShowDepthUncertainty_;
    pangolin::Var<bool>* isShowImages_; 
    bool hasShowImages_;


    pangolin::View disRecCam_;
    // pangolin::View disDepCam_;
    pangolin::OpenGlRenderState s_reccam_;
    // pangolin::OpenGlRenderState s_depcam_;

    pangolin::View disRefImage_;
    pangolin::View disRefDepth_;
    pangolin::View disTarImage_;

    pangolin::GlTexture texRefImage_;
    pangolin::GlTexture texRefDepth_;
    pangolin::GlTexture texTarImage_;



};
    
} // namespace MVS

