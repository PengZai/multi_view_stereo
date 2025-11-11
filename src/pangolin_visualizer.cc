#include "pangolin_visualizer.h"




namespace MVS
{

PangolinVisualizer::PangolinVisualizer(Config* const config):
Visualizer(config),
isProcessNextTarImage_(nullptr),
isShowImages_(nullptr),
hasShowImages_(false)
{

    float viewer_width = 960;
    float viewer_height = 600;
	const int UI_WIDTH = 180;

    pangolin::CreateWindowAndBind("Reconstruction Viewer", 2*viewer_width, 2*viewer_height);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);



   s_reccam_ =  pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(viewer_width,viewer_height,500,500,viewer_width/2,viewer_height/2,0.1,1000),
        pangolin::ModelViewLookAt(-10,-2,-10, 0,0,0, pangolin::AxisNegY)
    );

 
	// pangolin::View& disRefDepth = pangolin::Display("imgRefDepth")
	//     .SetAspect(image_window_width_/(float)image_window_height_);
    // float aspect = image_window_width_ / (float)image_window_height_;

    // disRefImage_ = pangolin::Display("disRefImage").SetAspect(aspect);
    // disTarImage_ = pangolin::Display("disTarImage").SetAspect(aspect);
    // disRefDepth_ = pangolin::Display("disRefDepth").SetAspect(aspect);


	// texRefImage_ = pangolin::GlTexture(image_window_width_,image_window_height_,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
	// texTarImage_ = pangolin::GlTexture(image_window_width_,image_window_height_,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
	// texRefDepth_ = pangolin::GlTexture(image_window_width_,image_window_height_,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);


    disRecCam_ = pangolin::CreateDisplay()
        .SetBounds(0.0,1.0,pangolin::Attach::Pix(UI_WIDTH),1.0,-viewer_width/viewer_height)
        .SetHandler(new pangolin::Handler3D(this->s_reccam_));

    // pangolin::CreateDisplay()
	// 	  .SetBounds(0.0, 0.33, pangolin::Attach::Pix(UI_WIDTH), 1.0)
    //       .SetLayout(pangolin::LayoutEqual)
    //       .AddDisplay(disRefImage_)
    //       .AddDisplay(disTarImage_)
    //       .AddDisplay(disRefDepth_);

		//   .AddDisplay(disRefDepth)

	// pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));
    isShowImages_ = new pangolin::Var<bool>("menu.showImages",true, true);
    isShowDepthUncertainty_ = new pangolin::Var<bool>("menu.showDepthUncertainty", true, true);
    isProcessNextTarImage_ = new pangolin::Var<bool>("menu.nextTarImage", false);

    // pangolin::CreateWindowAndBind("Depth uncertainty Viewer", 2*viewer_width, 2*viewer_height);

    // s_depcam_ =  pangolin::OpenGlRenderState(
    //     pangolin::ProjectionMatrix(viewer_width,viewer_height,500,500,viewer_width/2,viewer_height/2,0.1,1000),
    //     pangolin::ModelViewLookAt(5,5,10, 0,0,0, pangolin::AxisNegY)
    // );

    // disDepCam_ = pangolin::CreateDisplay()
    //     .SetBounds(0.0,1.0,pangolin::Attach::Pix(UI_WIDTH),1.0,-viewer_width/viewer_height)
    //     .SetHandler(new pangolin::Handler3D(this->s_depcam_));

}

PangolinVisualizer::~PangolinVisualizer()
{

}


void PangolinVisualizer::showRefImageReconstruction(Image* const ref_image, Image* const tar_image)
{

    std::vector<Image*> images = dataset_->getImages();
    int height = ref_image->getHeight();
    int width = ref_image->getWidth();
    int wh = width*height;
    Eigen::Matrix3f K_cam0 = config_->cameras_[0].getIntrinsicsMatrix();

    // if(*isShowImages_){
    //     uint8_t* ptr_ref_gray_rgb_data = ref_image->getGrayRGBDataPtr();
    //     uint8_t* ptr_tar_gray_rgb_data = tar_image->getGrayRGBDataPtr();
    //     float* ptr_ref_depth_data = ref_image->getDepthPtr();
    //     uint8_t* ptr_ref_depth_rgb_data = new uint8_t[3*width*height]();
    //     colorizedDepth(ptr_ref_depth_data, wh, ptr_ref_depth_rgb_data);

    //     texRefImage_.Upload(ptr_ref_gray_rgb_data, GL_RGB,GL_UNSIGNED_BYTE);
    //     texTarImage_.Upload(ptr_tar_gray_rgb_data, GL_RGB,GL_UNSIGNED_BYTE);
    //     texRefDepth_.Upload(ptr_ref_depth_rgb_data, GL_RGB,GL_UNSIGNED_BYTE);

    //     delete ptr_ref_depth_rgb_data;

    // }



    while (!pangolin::ShouldQuit()) {
        pangolin::BindToContext("Reconstruction Viewer");
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        disRecCam_.Activate(s_reccam_);
        // glClearColor(1.0f,1.0f,1.0f,1.0f);
        pangolin::glDrawAxis(0.5);
        
        if(*isProcessNextTarImage_){
            *isProcessNextTarImage_ = false;
            break;
        }
        
        Eigen::Matrix4f T_first_ref_image_world = MVS::invertTransform(ref_image->getTransformationMatrix());

        for(size_t i=0;i < images.size(); i++)
        {   

            // std::cout << "hello" << i << std::endl;

            MVS::Image* image = images[i];
            int image_pose_id =  image->getPoseId();
            int cam_id = image->getCameraId();

            Eigen::Matrix4f T_world_image = T_first_ref_image_world * image->getTransformationMatrix();

            if(image_pose_id == ref_image->getPoseId() && cam_id == ref_image->getCameraId()){
                Eigen::Vector4i color_rgba = Eigen::Vector4i(0,255,0,255);
                drawFrame(T_world_image, color_rgba, true, std::to_string(image->getPoseId()));

            }
            else if(image_pose_id == tar_image->getPoseId() && cam_id == tar_image->getCameraId()){
                Eigen::Vector4i color_rgba = Eigen::Vector4i(255,0,0,255);
                drawFrame(T_world_image, color_rgba, true, std::to_string(image->getPoseId()));
            }
            else{
                Eigen::Vector4i color_rgba = Eigen::Vector4i(0,0,255,25);
                drawFrame(T_world_image, color_rgba, false, std::to_string(image->getPoseId()));
            }
        }


  
        
        // float* min_depth_ptr = ref_image->getMinDepthPtr();
        // float* max_depth_ptr = ref_image->getMaxDepthPtr();
        Eigen::Matrix4f T_world_ref_image = T_first_ref_image_world * ref_image->getTransformationMatrix();

  

        PixelPoint* ref_ptr_pixel_point_matrix = ref_image->getPixelPointMatrixPtr();
        for(int v=0;v<height;v++)
        {
            for(int u=0;u<width;u++)
            {   
                int coord = v*width+u;
                float z = ref_ptr_pixel_point_matrix[coord].depth_;
                if(z < config_->min_depth_ || z > config_->max_depth_)
                {
                    continue;
                }
                float x = (u - K_cam0(0,2)) * z / K_cam0(0,0);
                float y = (v - K_cam0(1,2)) * z / K_cam0(1,1);

                float intensity = ref_ptr_pixel_point_matrix[coord].intensity_;   
                Eigen::Vector4f pc_h = Eigen::Vector4f(x,y,z, 1.0f); 
                Eigen::Vector4f pw_h = T_world_ref_image * pc_h;     
                Eigen::Vector3f pw = pw_h.head<3>() / pw_h[3];   

                drawPoint(pw, intensity);
            }
        }

        pangolin::FinishFrame();

       

        if(*isShowImages_)
        {
            if(hasShowImages_==false){
                
                hasShowImages_ = true;
                // showColorizedMinMaxDiffDepth(min_depth_ptr, max_depth_ptr, width, height, config_->min_depth_, config_->max_depth_, "ref_diff_depth");
                
                cv::Mat colorized_depth(height, width, CV_8UC3);
                ColorizedCVDepth(ref_ptr_pixel_point_matrix, width, height, config_->min_depth_, config_->max_depth_, colorized_depth);
                cv::imshow("ref_depth", colorized_depth);

                // showGrayImage(ptr_gray_data, width, height, "ref_image");
                // showGrayImage(tar_image->getGrayDataPtr(), width, height, "tar_image");

                if(*isShowDepthUncertainty_){


                }
     
            }

        }
        else{
            cv::destroyAllWindows();
            hasShowImages_ = false;

        }

        cv::waitKey(1);



    }
}





pangolin::View& PangolinVisualizer::getPangolinViewer()
{
    return disRecCam_;
}

pangolin::OpenGlRenderState& PangolinVisualizer::getPangolinRenderState()
{
    return s_reccam_;
}


void PangolinVisualizer::drawFrame(const Eigen::Matrix4f &T_w_c, const Eigen::Vector4i &bgra, bool drawAxis, const std::string &text)
{


    const float w = 0.1;
    const float h = w;
    const float z = 2*w;
    const float frame_line_width = 2.0;


    glPushMatrix();

    glMultMatrixf((GLfloat*)T_w_c.data());

    glPointSize(10.0f);  // Set point size in pixels
    glBegin(GL_POINTS);
    glColor4f(bgra[2]/255.0f,bgra[1]/255.0f,bgra[0]/255.0f, bgra[3]/255.0f);   // Set point color (red)
    glVertex3f(0, 0, 0);  // Plot point at (x=0, y=0, z=0)
    glEnd();

    // Draw axis, red - x green - y blue -z
    if(drawAxis){
        pangolin::glDrawAxis(w);
    }

    glLineWidth(frame_line_width);
    glColor4f(bgra[2]/255.0f,bgra[1]/255.0f,bgra[0]/255.0f, bgra[3]/255.0f);
    glBegin(GL_LINES);

    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);
    
    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);
    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);
    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);
    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glColor4f(bgra[2]/255.0f,bgra[1]/255.0f,bgra[0]/255.0f, bgra[3]/255.0f); // RGBA
    glBegin(GL_QUADS);
    glVertex3f(-w, -h, z); // bottom-left
    glVertex3f(w, -h, z);  // bottom-right
    glVertex3f(w, h, z);   // top-right
    glVertex3f(-w, h, z);  // top-left
    glEnd();



    // 🔹 Draw text label
    // pangolin::GlText& font = pangolin::GlText();
    // glColor4f(bgra[2]/255.0f,bgra[1]/255.0f,bgra[0]/255.0f, bgra[3]/255.0f); // red text
    // font.Text(text).Draw(0, 0, 0.02);  // slightly in front of frame
    pangolin::default_font().Text(text).Draw(0, 0, 0.02);

    glPopMatrix();


}


void PangolinVisualizer::drawPoint(const Eigen::Vector3f &pt3f, const Eigen::Vector3i &bgr){

    float point_size = 2.0;
    glPointSize(point_size);
    glBegin(GL_POINTS);
    glVertex3f(pt3f[0],pt3f[1],pt3f[2]);
    glColor3f(bgr[2]/255.0f,bgr[1]/255.0f,bgr[0]/255.0f);
    glEnd();

}

void PangolinVisualizer::drawPoint(const Eigen::Vector3f &pt3f, float gray)
{
    float point_size = 2.0f;
    glPointSize(point_size);

    // Normalize intensity if necessary
    float c = gray;                // could be 0–255 or 0–1
    if (c > 1.0f) c /= 255.0f;     // normalize to [0,1]
    c = std::clamp(c, 0.0f, 1.0f); // ensure valid range

    // Draw point
    glBegin(GL_POINTS);
    glColor3f(c, c, c);            // same intensity for R, G, B
    glVertex3f(pt3f[0], pt3f[1], pt3f[2]);
    glEnd();
}


} // namespace MVS



