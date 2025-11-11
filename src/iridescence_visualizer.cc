#include "iridescence_visualizer.h"


namespace MVS
{
    

CameraModel::CameraModel(const float size, const Eigen::Vector4f& color)
{
  float w = 1.0f * size;
  float h = 1.0f * size;
  float z = 1.0f * size;
  float alpha = color[3];


  

  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> coord_x_axis_vertices = {
    Eigen::Vector3f::Zero(),
    Eigen::Vector3f::UnitX() * 0.25
  };

  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> coord_y_axis_vertices = {
    Eigen::Vector3f::Zero(),
    Eigen::Vector3f::UnitY() * 0.25,
  };

  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> coord_z_axis_vertices = {
    Eigen::Vector3f::Zero(),
    Eigen::Vector3f::UnitZ() * 0.25,
  };




  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> backbone_vertices = {
    Eigen::Vector3f::Zero(),
    Eigen::Vector3f(w / 2, h / 2, z),
    Eigen::Vector3f::Zero(),
    Eigen::Vector3f(-w / 2, h / 2, z),
    Eigen::Vector3f::Zero(),
    Eigen::Vector3f(-w / 2, -h / 2, z),
    Eigen::Vector3f::Zero(),
    Eigen::Vector3f(w / 2, -h / 2, z),
    Eigen::Vector3f(w / 2, h / 2, z),
    Eigen::Vector3f(-w / 2, h / 2, z),
    Eigen::Vector3f(-w / 2, h / 2, z),
    Eigen::Vector3f(-w / 2, -h / 2, z),
    Eigen::Vector3f(-w / 2, -h / 2, z),
    Eigen::Vector3f(w / 2, -h / 2, z),
    Eigen::Vector3f(w / 2, -h / 2, z),
    Eigen::Vector3f(w / 2, h / 2, z)
  };

  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> frontside_vertices = {
        {w / 2, -h / 2, z},
        { -w / 2, -h / 2, z},
        { -w / 2, h / 2, z},
        { w / 2, h / 2, z},
  };





    std::vector<unsigned int> frontside_indices = { 
        1,2,0,
        0,2,3 
    };


  backbone_ = std::make_shared<glk::ThinLines>(backbone_vertices);
  backbone_shader_ = std::make_shared<guik::ShaderSetting>(guik::ColorMode::FLAT_COLOR);
  backbone_shader_->set_color(color);

  float coords_width = 0.01f;
  coord_x_axis_shader_  = std::make_shared<guik::ShaderSetting>(guik::ColorMode::FLAT_COLOR);
  coord_x_axis_shader_->set_color(Eigen::Vector4f(1.0f, 0.0f, 0.0f, color[3]));
  coord_x_axis_ = std::make_shared<glk::Lines>(coords_width, coord_x_axis_vertices);


  coord_y_axis_shader_  = std::make_shared<guik::ShaderSetting>(guik::ColorMode::FLAT_COLOR);
  coord_y_axis_shader_->set_color(Eigen::Vector4f(0.0f, 1.0f, 0.0f, color[3]));
  coord_y_axis_ = std::make_shared<glk::Lines>(coords_width, coord_y_axis_vertices);

  coord_z_axis_shader_  = std::make_shared<guik::ShaderSetting>(guik::ColorMode::FLAT_COLOR);
  coord_z_axis_shader_->set_color(Eigen::Vector4f(0.0f, 0.0f, 1.0f, color[3]));
  coord_z_axis_ = std::make_shared<glk::Lines>(coords_width, coord_z_axis_vertices);


  frontside_ = std::make_shared<glk::Mesh>(
        frontside_vertices.data(),             sizeof(float)*3,   // positions
        nullptr,                     0,                 // normals (none)
        nullptr,                     0,                 // colors (none)
        nullptr,                     0,                 // texcoords (none)
        static_cast<int>(frontside_vertices.size()),
        frontside_indices.data(),
        static_cast<int>(frontside_indices.size()),
        false/*wireframe=*/
    );

  frontside_shader_ = std::make_shared<guik::ShaderSetting>(guik::ColorMode::FLAT_COLOR);
  frontside_shader_->set_color(color);


}






CameraModel::~CameraModel()
{

}


void CameraModel::draw(guik::LightViewer* viewer, std::string name) const
{

  viewer->update_drawable("camera_backbone_" + name, backbone_, *backbone_shader_);
  viewer->update_drawable("camera_frontside_" + name, frontside_, *frontside_shader_);
  viewer->update_drawable("camera_coord_x_axis_" + name, coord_x_axis_, *coord_x_axis_shader_);
  viewer->update_drawable("camera_coord_y_axis_" + name, coord_y_axis_, *coord_y_axis_shader_);
  viewer->update_drawable("camera_coord_z_axis_" + name, coord_z_axis_, *coord_z_axis_shader_);

}




void CameraModel::setTransform(const Eigen::Matrix4f& transform)
{
  backbone_shader_->set_model_matrix(transform);
  frontside_shader_->set_model_matrix(transform);
  coord_x_axis_shader_->set_model_matrix(transform);
  coord_y_axis_shader_->set_model_matrix(transform);
  coord_z_axis_shader_->set_model_matrix(transform);


}

void CameraModel::setColor(const Eigen::Vector4f& color)
{
  backbone_shader_->set_color(color);
  frontside_shader_->set_color(color);
  coord_x_axis_shader_->set_alpha(color[3]);
  coord_y_axis_shader_->set_alpha(color[3]);
  coord_z_axis_shader_->set_alpha(color[3]);
}




IridescenceVisualizer::IridescenceVisualizer(Config* const config):
Visualizer(config),
viewer_(nullptr),
depth_uncertainty_viewer_(nullptr)
{

    float width = config_->cameras_[0].resolution_[0];
    float height = config_->cameras_[0].resolution_[1];

    picked_uncertainty_point_position_ = Eigen::Vector3f(0.0f, 0.0f, 0.0f);


    viewer_ = guik::LightViewer::instance(Eigen::Vector2i(1440, 960));
    depth_uncertainty_viewer_ = viewer_->sub_viewer("depth_uncertainty", Eigen::Vector2i(640, 480));

    viewer_->update_drawable("coord_system", glk::Primitives::coordinate_system(), guik::VertexColor());
    depth_uncertainty_viewer_->update_drawable("coord_system", glk::Primitives::coordinate_system(), guik::VertexColor());


    // Basic plotting
    if(config_->debug_plot_ == true)
    {
        viewer_->setup_plot("curves", 1024, 256);

    }
    // viewer_->update_plot_stairs("curves_y", "sin_stairs", ys);


    // viewer_->setup_plot("group02/circle", 1024, 256, ImPlotFlags_Equal);
    // viewer_->update_plot_line("group02/circle", "circle", xs, ys, ImPlotLineFlags_Loop);


    Tcl_ <<
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f,-1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f;


    viewer_->register_ui_callback("ui_callback", [&]() {
    // In the callback, you can call ImGui commands to create your UI.


        if (ImGui::Button("Close")) {
            viewer_->close();
            std::exit(0);
        }

        if (ImGui::Button("NextTarImage")) {

            isProcessNextTarImage_  = true;

        }

        auto& io = ImGui::GetIO();
        // If right clicked the GL canvas
        if (!io.WantCaptureMouse && io.MouseClicked[ImGuiMouseButton_Right]) {
            // Pick the depth of the clicked pixel
            float viewer_depth = viewer_->pick_depth({io.MousePos.x, io.MousePos.y});
            // If depth < 1.0f, the clicked pixel is a foreground object. Otherwise, it is the background.
            if(viewer_depth < 1.0f) {
                // Compute the 3D position of the clicked pixel
                Eigen::Vector3f pos = viewer_->unproject({io.MousePos.x, io.MousePos.y}, viewer_depth);
                std::cout << "viewer clicked_pos : " << pos[0] << "," << pos[1] << "," << pos[2] << std::endl;


                viewer_->update_drawable("sphere", glk::Primitives::wire_sphere(), guik::FlatColor(Eigen::Vector4f(1.0f, 0.0f, 0.0f, 0.5f)).translate(pos).scale(0.1));
            
                
            }

        }


    });




    depth_uncertainty_viewer_->register_ui_callback("depth_uncertainty_viewer_ui_callback", [=] {
        auto& io = ImGui::GetIO();

        // We would need some more conditions to check if the mouse is on the depth_uncertainty_viewer_ window
        if (io.MouseClicked[ImGuiMouseButton_Right]) {
            const auto winpos = ImGui::GetWindowPos();
            const Eigen::Vector2i clicked_pos = {io.MousePos.x - winpos.x, io.MousePos.y - winpos.y};

            const float depth = depth_uncertainty_viewer_->pick_depth(clicked_pos);
            if (depth > 0.0) {
                Eigen::Vector3f pos = depth_uncertainty_viewer_->unproject(clicked_pos, depth);

                std::cout << "depth_uncertainty_viewer clicked_pos : " << pos[0] << "," << pos[1] << "," << pos[2] << std::endl;

                picked_uncertainty_point_position_ = Eigen::Vector3f(round(pos[0]), round(pos[1]), 0.0f);
                depth_uncertainty_viewer_->update_drawable("sphere", glk::Primitives::wire_sphere(), 
                    guik::FlatColor(Eigen::Vector4f(1.0f, 0.0f, 0.0f, 0.5f)).translate(picked_uncertainty_point_position_).scale(1.0));

                DebugPlot();

            }
        }
    });



    std::vector<Eigen::Vector3f> ref_image_canvas_vertices = {

        { width, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
        { 0.0f,  height, 0.0f},
        { width, height, 0.0f}

    };

    std::vector<unsigned int> ref_image_canvas_indices = {
        1,2,0,
        0,2,3 
    };

    std::vector<Eigen::Vector2f> ref_image_canvas_tex_coords = {
        {1.0f, 0.0f},
        {0.0f, 0.0f},        
        {0.0f, 1.0f},
        {1.0f, 1.0f}
    };


    ref_image_canvas_ = std::make_shared<glk::Mesh>(
        ref_image_canvas_vertices.data(), sizeof(float)*3,
        nullptr, 0,   // vertex normal
        nullptr, 0,   // no colors
        ref_image_canvas_tex_coords.data(), sizeof(float) * 2,
        ref_image_canvas_vertices.size(),
        ref_image_canvas_indices.data(),
        ref_image_canvas_indices.size()
    );
    

}


void IridescenceVisualizer::DebugPlot()
{
    if(config_->debug_plot_ == true)
    {
        PixelPoint* ref_ptr_pixel_point_matrix = ref_image_->getPixelPointMatrixPtr();

        int ref_u = picked_uncertainty_point_position_[0];
        int ref_v = picked_uncertainty_point_position_[1];

        PixelPoint& picked_ref_ptr_pixel_point = ref_ptr_pixel_point_matrix[int(ref_v)* ref_image_->getWidth() + int(ref_u)];
        

        const cv::Mat& ref_cv_bgr_data = ref_image_->getBGRData();
        const cv::Mat& tar_cv_bgr_data = tar_image_->getBGRData();

        cv::Mat vis_cv_ref_rgb_data = ref_cv_bgr_data.clone();
        cv::Mat vis_cv_tar_rgb_data = tar_cv_bgr_data.clone();

        for(int min_max_idx=0;  min_max_idx < picked_ref_ptr_pixel_point.debug_info_vec_.size(); min_max_idx++ )
        {  

            DebugInfo& debug_info =  picked_ref_ptr_pixel_point.debug_info_vec_[min_max_idx];
            std::cout << "debug_info.costs " << debug_info.costs_.size() << std::endl;
            if(debug_info.costs_.size() == 0)
            {
                continue;
            }
            viewer_->update_plot_line("curves", "costs" + std::to_string(min_max_idx), debug_info.steps_, debug_info.costs_);  // When only Y values are given, X values become index IDs


            cv::circle(vis_cv_ref_rgb_data, cv::Point2i(ref_u,ref_v), 3, cv::Scalar(0,0,255), 2);
            cv::putText(vis_cv_ref_rgb_data, "("+std::to_string(ref_u)+","+std::to_string(ref_v)+")",cv::Point2i(ref_u,ref_v), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,0,255), 1.8);

            cv::Point2i cv_uv_best(round(debug_info.uv_best_match_[0]),round(debug_info.uv_best_match_[1]));
            cv::Point2i cv_uv_start(round(debug_info.valid_uvs_.front()[0]),round(debug_info.valid_uvs_.front()[1]));
            cv::Point2i cv_uv_end(round(debug_info.valid_uvs_.back()[0]),round(debug_info.valid_uvs_.back()[1]));

            cv::circle(vis_cv_tar_rgb_data, cv_uv_best, 3, cv::Scalar(255,0,255), 2); // pink
            cv::circle(vis_cv_tar_rgb_data, cv_uv_start, 2, cv::Scalar(0,0,255), 2); // red
            cv::putText(vis_cv_tar_rgb_data, "S",cv_uv_start, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 1.8);
            cv::line(vis_cv_tar_rgb_data, cv_uv_start, cv_uv_end, cv::Scalar(0,255,0), 1); // green
            cv::circle(vis_cv_tar_rgb_data, cv_uv_end, 2, cv::Scalar(255,0,0), 2); // blue
            cv::putText(vis_cv_tar_rgb_data, "E",cv_uv_end, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,0,0), 1.8);


        }


        std::shared_ptr<glk::Texture> ref_image_texture = glk::create_texture(vis_cv_ref_rgb_data);
        std::shared_ptr<glk::Texture> tar_image_texture = glk::create_texture(vis_cv_tar_rgb_data);

        viewer_->update_image("images/ref_image", ref_image_texture, 0.7);
        viewer_->update_image("images/tar_image", tar_image_texture, 0.7);
    }
}


void IridescenceVisualizer::showRefImageReconstruction(Image* const ref_image, Image* const tar_image)
{
    

    ref_image_ = ref_image;
    tar_image_ = tar_image;

    std::vector<Image*> images = dataset_->getImages();
    int height = ref_image->getHeight();
    int width = ref_image->getWidth();
    Eigen::Matrix3f K_cam0 = config_->cameras_[0].getIntrinsicsMatrix();
    Eigen::Matrix4f T_first_ref_image_world = MVS::invertTransform(ref_image->getTransformationMatrix());
    Eigen::Matrix4f T_world_ref_image = T_first_ref_image_world * ref_image->getTransformationMatrix();

    const cv::Mat& ref_cv_bgr_data = ref_image->getBGRData();
    const cv::Mat& tar_cv_bgr_data = tar_image->getBGRData();


    PixelPoint* ref_ptr_pixel_point_matrix = ref_image->getPixelPointMatrixPtr();


    cv::Mat ref_colorized_depth(height, width, CV_8UC3);
    ColorizedCVDepth(ref_ptr_pixel_point_matrix, width, height, config_->min_depth_, config_->max_depth_, ref_colorized_depth);

    std::shared_ptr<glk::Texture> ref_image_texture = glk::create_texture(ref_cv_bgr_data);
    std::shared_ptr<glk::Texture> ref_colorized_depth_texture = glk::create_texture(ref_colorized_depth);
    std::shared_ptr<glk::Texture> tar_image_texture = glk::create_texture(tar_cv_bgr_data);
    ref_image_canvas_->set_texture(ref_image_texture);

    viewer_->update_image("images/ref_image", ref_image_texture, 0.5);
    viewer_->update_image("images/ref_colorized_depth_image", ref_colorized_depth_texture, 0.5);
    viewer_->update_image("images/tar_image", tar_image_texture, 0.5);


    depth_uncertainty_viewer_->update_drawable("ref_image_canvas_", ref_image_canvas_, guik::TextureColor());


    for(size_t i=0;i < images.size(); i++)
    {   
        MVS::Image* image = images[i];
        int image_pose_id =  image->getPoseId();
        int cam_id = image->getCameraId();

        Eigen::Matrix4f T_world_image = Tcl_ * T_first_ref_image_world * image->getTransformationMatrix();

        std::shared_ptr<CameraModel> camera_model = std::make_shared<CameraModel>(0.3f, Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f));

        camera_model->setTransform(T_world_image);

        if(image_pose_id == ref_image->getPoseId() && cam_id == ref_image->getCameraId()){
            camera_model->setColor(Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f));

        }
        else if(image_pose_id == tar_image->getPoseId() && cam_id == tar_image->getCameraId()){
            camera_model->setColor(Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));
        }
        else{
            camera_model->setColor(Eigen::Vector4f(0.0f, 0.0f, 1.0f, 0.3f));
        }

        camera_model->draw(viewer_, "pose_"+std::to_string(image_pose_id)+"_camera_"+std::to_string(cam_id));


    }


    std::vector<Eigen::Vector3f> point_cloud_vertices;
    std::vector<Eigen::Vector4f> point_cloud_colors;
    std::vector<Eigen::Vector3f> depth_uncertainty_vertices;




    for(int v=0;v<height;v++)
    {
        for(int u=0;u<width;u++)
        {   
            int coord = v*width+u;
            float z = ref_ptr_pixel_point_matrix[coord].depth_;
            float min_z = 1/ref_ptr_pixel_point_matrix[coord].max_inv_depth_vec_[0];
            float max_z = 1/ref_ptr_pixel_point_matrix[coord].min_inv_depth_vec_[0];
            float diff_z = max_z - min_z;
            if(diff_z > 100)
            {
                diff_z = 100;
            }

            depth_uncertainty_vertices.push_back(Eigen::Vector3f(u,v,0.0f));
            depth_uncertainty_vertices.push_back(Eigen::Vector3f(u,v,diff_z));

            if(max_z - min_z > 1.0f || z < config_->min_depth_ || z > config_->max_depth_)
            {
                continue;
            }
            float x = (u - K_cam0(0,2)) * z / K_cam0(0,0);
            float y = (v - K_cam0(1,2)) * z / K_cam0(1,1);

            // const cv::Vec3b& pix_bgr = ref_cv_bgr_data.at<cv::Vec3b>(v, u);
            const float intensity = ref_ptr_pixel_point_matrix[coord].intensity_;
            Eigen::Vector4f pc_h = Eigen::Vector4f(x,y,z, 1.0f); 
            Eigen::Vector4f pw_h = T_world_ref_image * pc_h;
            pw_h = Tcl_*pw_h;     
            Eigen::Vector3f pw = pw_h.head<3>() / pw_h[3]; 

            point_cloud_vertices.emplace_back(pw);
            point_cloud_colors.emplace_back(Eigen::Vector4f(intensity/255.0f, intensity/255.0f, intensity/255.0f, 1.0f));
            // std::cout << "v,u:" << v << "," << u << "." << pix_bgr << "," << intensity << std::endl;;

     

        }
    }

    std::shared_ptr<glk::PointCloudBuffer> point_cloud_buffer_ptr = std::make_shared<glk::PointCloudBuffer>(point_cloud_vertices);
    point_cloud_buffer_ptr->add_color(point_cloud_colors);
    std::shared_ptr<guik::ShaderSetting> point_cloud_buffer_shader_setting = std::make_shared<guik::ShaderSetting>(guik::ColorMode::VERTEX_COLOR);
    point_cloud_buffer_shader_setting->set_point_scale(1.5f);
    viewer_->update_drawable("points", point_cloud_buffer_ptr, *point_cloud_buffer_shader_setting);

    depth_uncertainty_viewer_->update_drawable("depth uncertainty grid", std::make_shared<glk::ThinLines>(depth_uncertainty_vertices), guik::FlatColor(0.0f, 1.0f, 0.0f, 0.5f));

    DebugPlot();


    while (viewer_->spin_once()) {
                
        if(isProcessNextTarImage_  == true)
        {
            isProcessNextTarImage_  = false;
            break;
        }
    }

}


} // namespace MVS


