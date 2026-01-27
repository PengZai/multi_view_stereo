#include "iridescence_visualizer.h"
#include "multi_view_stereo.h"


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


    // picked_uncertainty_point_position_ = Eigen::Vector3f(41.0f, 4.0f, 0.0f);

    // picked_uncertainty_point_position_ = Eigen::Vector3f(101.0f, 229.0f, 0.0f);

    picked_uncertainty_point_position_ = Eigen::Vector3f(752.0f, 330.0f, 0.0f);

    // viewer_ = guik::LightViewer::instance(Eigen::Vector2i(1440, 960));
    viewer_ = guik::LightViewer::instance(Eigen::Vector2i(-1, -1));

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


    viewer_->register_drawable_filter("drawable_filter", [&](const std::string& drawable_name) {
        if(!(selected_idx_display_mode_==0) )
        {


        }

        if(!(selected_idx_display_mode_==1)&& drawable_name.find("point-") != std::string::npos)
        {
            Image* const ref_image = multi_view_stereo_->getReferenceImage();        
            if(ref_image == nullptr)
            {
                return false;
            }

            std::string point_label = "point-" + std::to_string(ref_image->getId());

            if(drawable_name == point_label)
            {
                return true;             
            }
            else{
                return false;
            }

        }
    


        return true;
    });
    
    viewer_->register_ui_callback("ui_callback", [&]() {
    // In the callback, you can call ImGui commands to create your UI.
        
        if (ImGui::Button("Close")) 
        {
            viewer_->close();
            std::exit(0);
        }

        MultiViewStereo::Status multi_view_stereo_status = multi_view_stereo_->getStatus();

        std::string number_of_trajectory_label = "MVS status:" + MultiViewStereo::StatusToString(multi_view_stereo_status) + ", Minimum traj:" + std::to_string(config_->minimum_traj_) + ", Maximum traj:" + std::to_string(config_->maximum_traj_);
        ImGui::Text(number_of_trajectory_label.c_str());
        if(ImGui::TreeNode("Common"))
        {
            
            const char* display_modes[] = {"select reference frame depth", "scene reconstruction"};
            if(ImGui::Combo("display mode", &selected_idx_display_mode_, display_modes, IM_ARRAYSIZE(display_modes)))
            {
                isNextAction_  = true;
            }

            if(ImGui::Checkbox("is show on consistency check", &isShowOnConsistencycheck_))
            {
                isNextAction_  = true;
            }
            
            if (ImGui::Button("Consistency Check")) 
            {
                multi_view_stereo_->consistency_check();
            }

            ImGui::TreePop();
        }

        if(ImGui::TreeNode("Single Depth Reconstruction"))
        {
            static bool is_save_debug_info = true;
            if(ImGui::Checkbox("isSaveDebuginfo", &is_save_debug_info))
            {

            }

            
            if(ImGui::InputInt("selected_ref_image_id_", &selected_ref_image_id_))
            {
                NextRefImageEvent();

            }
            
            // int test = 10;
            // if(ImGui::InputInt("test", &test))
            // {

            // }

            if(ImGui::InputInt("selected_tar_image_id_", &selected_tar_image_id_))
            {
                NextTarImageEvent();
            }

            if (ImGui::Button("Single Depth Reconstruction")) 
            {

                multi_view_stereo_->setIsRunSingleDepthReconstruction(true);
                isNextAction_  = true;

            }


            ImGui::SameLine();
            static int counter = 0;
            float spacing = ImGui::GetStyle().ItemInnerSpacing.x;
            if (ImGui::ArrowButton("##left", ImGuiDir_Left)) { selected_tar_image_id_--; NextTarImageEvent(); }
            ImGui::SameLine(0.0f, spacing);
            if (ImGui::ArrowButton("##right", ImGuiDir_Right)) { selected_tar_image_id_++; NextTarImageEvent(); }
            ImGui::SameLine();
            ImGui::Text("Next/Previous selected tar frame: %d", counter);
           

            ImGui::TreePop();
        }

        if(ImGui::TreeNode("Scene Reconstruction"))
        {
        

            static int min_frame_id = 0;
            ImGui::InputInt("min_frame_id", &min_frame_id);

            static int max_frame_id = 20;
            ImGui::InputInt("max_frame_id", &max_frame_id);
            
            if (ImGui::Button("Scene Reconstruction")) 
            {
                
                multi_view_stereo_->setIsRunSceneReconstruction(true);
                multi_view_stereo_->scene_reconstruction_init();

                isNextAction_  = true;

            }

            ImGui::TreePop();
        }




        if(ptr_picked_ref_ptr_pixel_point_ != nullptr)
        {
            DebugInfo& debug_info = ptr_picked_ref_ptr_pixel_point_->debug_info_;

            ImGui::Begin("Cost Curve");
            const std::string pixel_point_status_label = "PixelPoint Status: " + PixelPoint::StatusToString(ptr_picked_ref_ptr_pixel_point_->status_) + 
                ", num aggregation : " + std::to_string(ptr_picked_ref_ptr_pixel_point_->num_has_been_aggregated_) + 
                ", num consistency check: " + std::to_string(ptr_picked_ref_ptr_pixel_point_->num_correct_consistency_check_);
            ImGui::BulletText(pixel_point_status_label.c_str());
            ImGui::Indent();
            const std::string pixel_point_position_label = "PixelPoint u:" + std::to_string(ptr_picked_ref_ptr_pixel_point_->u_) + ",v:" + std::to_string(ptr_picked_ref_ptr_pixel_point_->v_) + ", epipolar length:" + std::to_string(debug_info.epipolar_length_);
            ImGui::Text(pixel_point_position_label.c_str());
            const std::string pixel_point_depth_uncertainty_label = "Depth Uncertainty (" + std::to_string(ptr_picked_ref_ptr_pixel_point_->min_depth_) + "," + std::to_string(ptr_picked_ref_ptr_pixel_point_->max_depth_) + ")";
            ImGui::Text(pixel_point_depth_uncertainty_label.c_str());

            if(ptr_picked_ref_ptr_pixel_point_->costs_.size() > 0){

                // std::cout << "debug info size:" << ptr_picked_ref_ptr_pixel_point_->debug_info_vec_.size() << ", " << ptr_picked_ref_ptr_pixel_point_->epipolar_segment_vec_[0].min_depth_ << "-" << ptr_picked_ref_ptr_pixel_point_->epipolar_segment_vec_[0].max_depth_ << std::endl;

                std::string manual_match_label = "manual match ";
                
                if(ImGui::SliderInt(manual_match_label.data(), &debug_info.manual_step_idx_, 0, debug_info.steps_.size()-1, "%d s"))
                {
                    // if we adjust slider

                    DebugPlot();
                    // std::cout << "ptr_picked_ref_ptr_pixel_point_->aggregate_costs_ start" << std::endl;
                    // for(int idx=0;idx < ptr_picked_ref_ptr_pixel_point_->aggregate_costs_.size();idx++){
                    //     std::cout << ptr_picked_ref_ptr_pixel_point_->aggregate_costs_[idx] << ",";
                    //     if(idx == ptr_picked_ref_ptr_pixel_point_->aggregate_costs_.size()-1){
                    //         std::cout<<std::endl;
                    //     }
                    // }
                    // std::cout << "ptr_picked_ref_ptr_pixel_point_->aggregate_costs_ end" << std::endl;
                }

                const std::string epipolar_segment_label = "Epipolar Segment Cost";

                if (ImPlot::BeginPlot(epipolar_segment_label.c_str())) {

                    ImPlot::SetupAxes("step","cost");
                    std::string cost_label = "cost ";
                    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
                    ImPlot::PlotLine(cost_label.data(), debug_info.steps_.data(), ptr_picked_ref_ptr_pixel_point_->costs_.data(), ptr_picked_ref_ptr_pixel_point_->costs_.size());
                    ImPlot::PlotInfLines(manual_match_label.data(), debug_info.steps_.data() + debug_info.manual_step_idx_, 1);

                    // std::string best_match_label = "best match " + std::to_string(debug_info_idx);
                    // ImPlot::PlotInfLines(best_match_label.data(), debug_info.steps_.data() + debug_info.best_step_idx_, 1);
                    // ImPlot::PlotInfLines(manual_match_label.data(), debug_info.steps_.data() + debug_info.manual_step_idx_, 1);

                    ImPlot::EndPlot();
                }

                if(ptr_picked_ref_ptr_pixel_point_->status_ != PixelPoint::Status::INVALID_COST_OBSERVATION)
                {
                    
                    const std::string epipolar_segment_label = "Epipolar Segment Aggregation Cost ";
                    if (ImPlot::BeginPlot(epipolar_segment_label.c_str())) {

                        ImPlot::SetupAxes("step","aggregated cost");
                        std::string aggregated_cost_label = "aggregated cost";
                        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
                        ImPlot::PlotLine(aggregated_cost_label.data(), debug_info.steps_.data(), ptr_picked_ref_ptr_pixel_point_->aggregate_costs_.data(), ptr_picked_ref_ptr_pixel_point_->aggregate_costs_.size());
                        // std::string best_match_label = "best match " + std::to_string(debug_info_idx);
                        // ImPlot::PlotInfLines(best_match_label.data(), debug_info.steps_.data() + debug_info.best_step_idx_, 1);
                        ImPlot::PlotInfLines(manual_match_label.data(), debug_info.steps_.data() + debug_info.manual_step_idx_, 1);

                        if(debug_info.mini_aggregate_cost_idx_ != -1){
                            
                            std::string mini_aggregate_cost_label = " mini_aggregate_cost ";
                            ImPlot::PlotInfLines(mini_aggregate_cost_label.data(), debug_info.steps_.data() + debug_info.mini_aggregate_cost_idx_, 1);
                            ImPlot::PlotInfLines(mini_aggregate_cost_label.data(), debug_info.steps_.data() + debug_info.mini_aggregate_cost_idx_plus_, 1);
                            ImPlot::PlotInfLines(mini_aggregate_cost_label.data(), debug_info.steps_.data() + debug_info.mini_aggregate_cost_idx_minus_, 1);

                        } 

                        ImPlot::EndPlot();

                    }

                    const std::string incremental_aggregate_cost_from_left_direction = "Incremental Aggregation Cost From Left Direction";
                    if (ImPlot::BeginPlot(incremental_aggregate_cost_from_left_direction.c_str())) {
                        ImPlot::SetupAxes("step","incremental aggregated cost");
                        std::string incremental_aggregated_cost_label = "incremental aggregated cost";
                        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
                        ImPlot::PlotLine(incremental_aggregated_cost_label.data(), debug_info.steps_.data(), debug_info.tmp_aggregate_costs_from_left_direction_.data(), debug_info.tmp_aggregate_costs_from_left_direction_.size());
                        ImPlot::PlotInfLines(manual_match_label.data(), debug_info.steps_.data() + debug_info.manual_step_idx_, 1);
                        ImPlot::EndPlot();
                    }

                    const std::string incremental_aggregate_cost_from_right_direction = "Incremental Aggregation Cost From Right Direction";
                    if (ImPlot::BeginPlot(incremental_aggregate_cost_from_right_direction.c_str())) {
                        ImPlot::SetupAxes("step","incremental aggregated cost");
                        std::string incremental_aggregated_cost_label = "incremental aggregated cost";
                        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
                        ImPlot::PlotLine(incremental_aggregated_cost_label.data(), debug_info.steps_.data(), debug_info.tmp_aggregate_costs_from_right_direction_.data(), debug_info.tmp_aggregate_costs_from_right_direction_.size());
                        ImPlot::PlotInfLines(manual_match_label.data(), debug_info.steps_.data() + debug_info.manual_step_idx_, 1);
                        ImPlot::EndPlot();
                    }

                    const std::string incremental_aggregate_cost_from_up_direction = "Incremental Aggregation Cost From Up Direction";
                    if (ImPlot::BeginPlot(incremental_aggregate_cost_from_up_direction.c_str())) {
                        ImPlot::SetupAxes("step","incremental aggregated cost");
                        std::string incremental_aggregated_cost_label = "incremental aggregated cost";
                        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
                        ImPlot::PlotLine(incremental_aggregated_cost_label.data(), debug_info.steps_.data(), debug_info.tmp_aggregate_costs_from_up_direction_.data(), debug_info.tmp_aggregate_costs_from_up_direction_.size());
                        ImPlot::PlotInfLines(manual_match_label.data(), debug_info.steps_.data() + debug_info.manual_step_idx_, 1);
                        ImPlot::EndPlot();
                    }

                    const std::string incremental_aggregate_cost_from_down_direction = "Incremental Aggregation Cost From Down Direction";
                    if (ImPlot::BeginPlot(incremental_aggregate_cost_from_down_direction.c_str())) {
                        ImPlot::SetupAxes("step","incremental aggregated cost");
                        std::string incremental_aggregated_cost_label = "incremental aggregated cost";
                        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
                        ImPlot::PlotLine(incremental_aggregate_cost_from_down_direction.data(), debug_info.steps_.data(), debug_info.tmp_aggregate_costs_from_down_direction_.data(), debug_info.tmp_aggregate_costs_from_down_direction_.size());
                        ImPlot::PlotInfLines(manual_match_label.data(), debug_info.steps_.data() + debug_info.manual_step_idx_, 1);
                        ImPlot::EndPlot();
                    }

                    if (ImPlot::BeginPlot("Depth Plots")) {

                        ImPlot::SetupAxes("step","value");
                        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
                        std::string depth_label = "depth ";
                        std::vector<float> depths(ptr_picked_ref_ptr_pixel_point_->inv_depths_.size());
                        for (size_t i = 0; i < depths.size(); ++i) {
                            depths[i] = 1.0/ptr_picked_ref_ptr_pixel_point_->inv_depths_[i];
                        }
                        ImPlot::PlotLine(depth_label.data(), debug_info.steps_.data(), depths.data(), depths.size());
                        ImPlot::PlotInfLines(manual_match_label.data(), debug_info.steps_.data() + debug_info.manual_step_idx_, 1);

                        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
                        std::string inv_depth_label = "inv_depth ";
                        ImPlot::PlotLine(inv_depth_label.data(), debug_info.steps_.data(), ptr_picked_ref_ptr_pixel_point_->inv_depths_.data(), ptr_picked_ref_ptr_pixel_point_->inv_depths_.size());
                        // std::string best_match_label = "best match " + std::to_string(epipolar_segment_idx);
                        // ImPlot::PlotInfLines(best_match_label.data(), debug_info.steps_.data() + debug_info.best_step_idx_, 1);

                        if(debug_info.mini_aggregate_cost_idx_ != -1){       
                            std::string mini_aggregate_cost_label = " mini_aggregate_cost ";
                            ImPlot::PlotInfLines(mini_aggregate_cost_label.data(), debug_info.steps_.data() + debug_info.mini_aggregate_cost_idx_, 1);
                        }



                        ImPlot::EndPlot();
                    }

                }
                else
                {
                    // std::cout << "the pixel " << 
                    //     ptr_picked_ref_ptr_pixel_point_->u_ << "," << ptr_picked_ref_ptr_pixel_point_->v_ << " is PixelPoint::Status::INVALID search within depth " << 1.0f/ptr_picked_ref_ptr_pixel_point_->max_inv_depth_ << "," << 1.0f/ptr_picked_ref_ptr_pixel_point_->min_inv_depth_ << std::endl;
                    
                
                }

                

            }
            
            ImGui::End();
        }

        if(ptr_pixel_point_on_the_left_ref_picked_pixel_point_ != nullptr)
        {
            DebugInfo& debug_info_on_the_left =  ptr_pixel_point_on_the_left_ref_picked_pixel_point_->debug_info_;

            ImGui::Begin("Aggregation Cost Plots on the Left");
            const std::string pixel_point_on_the_left_ref_picked_pixel_point_status_label = "PixelPoint Status: " + PixelPoint::StatusToString(ptr_pixel_point_on_the_left_ref_picked_pixel_point_->status_);
            ImGui::BulletText(pixel_point_on_the_left_ref_picked_pixel_point_status_label.c_str());
            ImGui::Indent();
            const std::string pixel_point_on_the_left_ref_picked_pixel_point_position_label = "Pixel Point u:" + std::to_string(ptr_pixel_point_on_the_left_ref_picked_pixel_point_->u_) + ",v:" + std::to_string(ptr_pixel_point_on_the_left_ref_picked_pixel_point_->v_);
            ImGui::Text(pixel_point_on_the_left_ref_picked_pixel_point_position_label.c_str());
            const std::string pixel_point_on_the_left_ref_picked_pixel_point_depth_uncertainty_label = "Depth Uncertainty (" + std::to_string(ptr_pixel_point_on_the_left_ref_picked_pixel_point_->min_depth_) + "," + std::to_string(ptr_pixel_point_on_the_left_ref_picked_pixel_point_->max_depth_) + ")";
            ImGui::Text(pixel_point_on_the_left_ref_picked_pixel_point_depth_uncertainty_label.c_str());

            if(ptr_pixel_point_on_the_left_ref_picked_pixel_point_->costs_.size() > 0)
            {
                
                std::string manual_match_label = "manual match ";
                if(ImGui::SliderInt(manual_match_label.data(), &debug_info_on_the_left.manual_step_idx_, 0, debug_info_on_the_left.steps_.size()-1, "%d s"))
                {
                    // if we adjust slider
                    DebugPlot();

                }

                const std::string epipolar_segment_label = "Epipolar Segment Cost";

                if (ImPlot::BeginPlot(epipolar_segment_label.c_str())) {

                    ImPlot::SetupAxes("step","cost");
                    std::string cost_label = "cost ";
                    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
                    ImPlot::PlotLine(cost_label.data(), debug_info_on_the_left.steps_.data(), ptr_pixel_point_on_the_left_ref_picked_pixel_point_->costs_.data(), ptr_pixel_point_on_the_left_ref_picked_pixel_point_->costs_.size());
                    ImPlot::PlotInfLines(manual_match_label.data(), debug_info_on_the_left.steps_.data() + debug_info_on_the_left.manual_step_idx_, 1);

                    // std::string best_match_label = "best match " + std::to_string(debug_info_idx);
                    // ImPlot::PlotInfLines(best_match_label.data(), debug_info.steps_.data() + debug_info.best_step_idx_, 1);
                    // ImPlot::PlotInfLines(manual_match_label.data(), debug_info.steps_.data() + debug_info.manual_step_idx_, 1);

                    ImPlot::EndPlot();
                }

                if(ptr_pixel_point_on_the_left_ref_picked_pixel_point_->status_ != PixelPoint::Status::INVALID_COST_OBSERVATION){

                    const std::string epipolar_segment_label = "Epipolar Segment Aggregation Cost ";
                    if (ImPlot::BeginPlot(epipolar_segment_label.c_str())) {
                        ImPlot::SetupAxes("step","aggregated cost");
                        std::string aggregated_cost_label = "aggregated cost ";
                        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
                        ImPlot::PlotLine(aggregated_cost_label.data(), debug_info_on_the_left.steps_.data(), ptr_pixel_point_on_the_left_ref_picked_pixel_point_->aggregate_costs_.data(), ptr_pixel_point_on_the_left_ref_picked_pixel_point_->aggregate_costs_.size());
                        // std::string best_match_label = "best match " + std::to_string(epipolar_segment_idx);
                        // ImPlot::PlotInfLines(best_match_label.data(), debug_info_on_the_left.steps_.data() + debug_info_on_the_left.best_step_idx_, 1);
                        ImPlot::PlotInfLines(manual_match_label.data(), debug_info_on_the_left.steps_.data() + debug_info_on_the_left.manual_step_idx_, 1);

                        if(debug_info_on_the_left.mini_aggregate_cost_idx_ != -1){       
                            std::string mini_aggregate_cost_label = " mini_aggregate_cost ";
                            ImPlot::PlotInfLines(mini_aggregate_cost_label.data(), debug_info_on_the_left.steps_.data() + debug_info_on_the_left.mini_aggregate_cost_idx_, 1);
                        }

                        ImPlot::EndPlot();
                    }

                }
                
                
            }
            ImGui::End();
        }

        if(ptr_pixel_point_on_the_right_ref_picked_pixel_point_ != nullptr)
        {
            DebugInfo& debug_info_on_the_right =  ptr_pixel_point_on_the_right_ref_picked_pixel_point_->debug_info_;

            ImGui::Begin("Aggregation Cost Plots on the Right");
            const std::string pixel_point_on_the_right_ref_picked_pixel_point_status_label = "PixelPoint Status: " + PixelPoint::StatusToString(ptr_pixel_point_on_the_right_ref_picked_pixel_point_->status_);
            ImGui::BulletText(pixel_point_on_the_right_ref_picked_pixel_point_status_label.c_str());
            ImGui::Indent();
            const std::string pixel_point_on_the_right_ref_picked_pixel_point_posistion_label = "Pixel Point u:" + std::to_string(ptr_pixel_point_on_the_right_ref_picked_pixel_point_->u_) + ",v:" + std::to_string(ptr_pixel_point_on_the_right_ref_picked_pixel_point_->v_);
            ImGui::Text(pixel_point_on_the_right_ref_picked_pixel_point_posistion_label.c_str());
            const std::string pixel_point_on_the_right_ref_picked_pixel_point_depth_uncertainty_label = "Depth Uncertainty (" + std::to_string(ptr_pixel_point_on_the_right_ref_picked_pixel_point_->min_depth_) + "," + std::to_string(ptr_pixel_point_on_the_right_ref_picked_pixel_point_->max_depth_) + ")";
            ImGui::Text(pixel_point_on_the_right_ref_picked_pixel_point_depth_uncertainty_label.c_str());

            if(ptr_pixel_point_on_the_right_ref_picked_pixel_point_->costs_.size() > 0)
            {
                
                std::string manual_match_label = "manual match ";
                if(ImGui::SliderInt(manual_match_label.data(), &debug_info_on_the_right.manual_step_idx_, 0, debug_info_on_the_right.steps_.size()-1, "%d s"))
                {
                    // if we adjust slider
                    DebugPlot();

                }

                const std::string epipolar_segment_label = "Epipolar Segment Cost";

                if (ImPlot::BeginPlot(epipolar_segment_label.c_str())) {

                    ImPlot::SetupAxes("step","cost");
                    std::string cost_label = "cost ";
                    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
                    ImPlot::PlotLine(cost_label.data(), debug_info_on_the_right.steps_.data(), ptr_pixel_point_on_the_right_ref_picked_pixel_point_->costs_.data(), ptr_pixel_point_on_the_right_ref_picked_pixel_point_->costs_.size());
                    ImPlot::PlotInfLines(manual_match_label.data(), debug_info_on_the_right.steps_.data() + debug_info_on_the_right.manual_step_idx_, 1);

                    // std::string best_match_label = "best match " + std::to_string(debug_info_idx);
                    // ImPlot::PlotInfLines(best_match_label.data(), debug_info.steps_.data() + debug_info.best_step_idx_, 1);
                    // ImPlot::PlotInfLines(manual_match_label.data(), debug_info.steps_.data() + debug_info.manual_step_idx_, 1);

                    ImPlot::EndPlot();
                }

                if(ptr_pixel_point_on_the_right_ref_picked_pixel_point_->status_ != PixelPoint::Status::INVALID_COST_OBSERVATION){

                    const std::string epipolar_segment_label = "Epipolar Segment Aggregation Cost ";
                    if (ImPlot::BeginPlot(epipolar_segment_label.c_str())) {
                        ImPlot::SetupAxes("step","aggregated cost");
                        std::string aggregated_cost_label = "aggregated cost ";
                        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
                        ImPlot::PlotLine(aggregated_cost_label.data(), debug_info_on_the_right.steps_.data(), ptr_pixel_point_on_the_right_ref_picked_pixel_point_->aggregate_costs_.data(), ptr_pixel_point_on_the_right_ref_picked_pixel_point_->aggregate_costs_.size());
                        // std::string best_match_label = "best match " + std::to_string(epipolar_segment_idx);
                        // ImPlot::PlotInfLines(best_match_label.data(), debug_info_on_the_right.steps_.data() + debug_info_on_the_right.best_step_idx_, 1);
                        ImPlot::PlotInfLines(manual_match_label.data(), debug_info_on_the_right.steps_.data() + debug_info_on_the_right.manual_step_idx_, 1);

                        if(debug_info_on_the_right.mini_aggregate_cost_idx_ != -1){       
                            std::string mini_aggregate_cost_label = " mini_aggregate_cost ";
                            ImPlot::PlotInfLines(mini_aggregate_cost_label.data(), debug_info_on_the_right.steps_.data() + debug_info_on_the_right.mini_aggregate_cost_idx_, 1);
                        }

                        ImPlot::EndPlot();
                    }

                }
                
                
            }
            ImGui::End();
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

                int picked_u = round(pos[0]);
                int picked_v = round(pos[1]);
                picked_uncertainty_point_position_ = Eigen::Vector3f(picked_u, picked_v, 0.0f);
                depth_uncertainty_viewer_->update_drawable("sphere", glk::Primitives::wire_sphere(), 
                    guik::FlatColor(Eigen::Vector4f(1.0f, 0.0f, 0.0f, 0.5f)).translate(picked_uncertainty_point_position_).scale(1.0));

                DebugPlot();

                if(ptr_picked_ref_ptr_pixel_point_ != nullptr)
                {
                    Image* const ref_image = multi_view_stereo_->getReferenceImage();
                    Image* const tar_image = multi_view_stereo_->getTargetImage();

                    int height = ref_image->getHeight();
                    int width = tar_image->getWidth();
                    Eigen::Matrix3f K_cam0 = config_->cameras_[0].getIntrinsicsMatrix();
                    Eigen::Matrix4f T_first_ref_image_world = MVS::invertTransform(ref_image->getTransformationMatrix());
                    Eigen::Matrix4f T_world_ref_image = T_first_ref_image_world * tar_image->getTransformationMatrix();

                    float z = ptr_picked_ref_ptr_pixel_point_->output_data_.depth_;
                    float x = (picked_u - K_cam0(0,2)) * z / K_cam0(0,0);
                    float y = (picked_v - K_cam0(1,2)) * z / K_cam0(1,1);

                    // const cv::Vec3b& pix_bgr = ref_cv_bgr_data.at<cv::Vec3b>(v, u);
                    Eigen::Vector4f pc_h = Eigen::Vector4f(x,y,z, 1.0f); 
                    Eigen::Vector4f pw_h = T_world_ref_image * pc_h;
                    pw_h = Tcl_*pw_h;     
                    Eigen::Vector3f picked_pw = pw_h.head<3>() / pw_h[3]; 

                    viewer_->update_drawable("sphere", glk::Primitives::wire_sphere(), guik::FlatColor(Eigen::Vector4f(1.0f, 0.0f, 0.0f, 0.5f)).translate(picked_pw).scale(0.1));

                }
                

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




void IridescenceVisualizer::NextRefImageEvent()
{
    // Image* ref_image_ =  multi_view_stereo_->getReferenceImage();

    std::vector<Image*>& images = dataset_->getImages();

    if(selected_ref_image_id_ == selected_tar_image_id_ && selected_ref_image_id_ < last_selected_ref_image_id_)
    {
        selected_ref_image_id_--;
    }
    else if(selected_ref_image_id_ == selected_tar_image_id_ && selected_ref_image_id_ > last_selected_ref_image_id_)
    {
        selected_ref_image_id_++;
    }

    if(selected_ref_image_id_ > images.size()-1)
    {
        selected_ref_image_id_ = 0;
    }
    if(selected_ref_image_id_ < 0)
    {
        selected_ref_image_id_ = images.size()-1;
    }

    last_selected_ref_image_id_ = selected_ref_image_id_;

    Image* new_ref_image = images[selected_ref_image_id_];
    multi_view_stereo_->setReferenceImage(new_ref_image);
    
    isNextAction_ = true;

}



void IridescenceVisualizer::NextTarImageEvent()
{

    std::vector<Image*>& images = dataset_->getImages();
    if(selected_tar_image_id_ == selected_ref_image_id_ && selected_tar_image_id_ < last_selected_tar_image_id_)
    {
        selected_tar_image_id_--;
    }
    else if(selected_tar_image_id_ == selected_ref_image_id_ && selected_tar_image_id_ > last_selected_tar_image_id_)
    {
        selected_tar_image_id_++;
    }

    if(selected_tar_image_id_ > images.size()-1)
    {
        selected_tar_image_id_ = 0;
    }
    if(selected_tar_image_id_ < 0)
    {
        selected_tar_image_id_ = images.size()-1;
    }

    last_selected_tar_image_id_ = selected_tar_image_id_;

    Image* new_tar_image = images[selected_tar_image_id_];
    multi_view_stereo_->setTargetImage(new_tar_image);

    isNextAction_ = true;

}


void IridescenceVisualizer::DebugPlot()
{
    if(config_->debug_plot_ == true)
    {
        Image* const ref_image = multi_view_stereo_->getReferenceImage();
        Image* const tar_image = multi_view_stereo_->getTargetImage();

        PixelPoint* ref_ptr_pixel_point_matrix = ref_image->getPixelPointMatrixPtr();

        int ref_u = picked_uncertainty_point_position_[0];
        int ref_v = picked_uncertainty_point_position_[1];
        int width = ref_image->getWidth();
        int height = ref_image->getHeight();

        ptr_picked_ref_ptr_pixel_point_ = ref_ptr_pixel_point_matrix + int(ref_v)* width + int(ref_u);
        if(ref_u - 1 >= 0)
        {
            ptr_pixel_point_on_the_left_ref_picked_pixel_point_ = ref_ptr_pixel_point_matrix + int(ref_v)* width + int(ref_u-1);
        }
        else{
            ptr_pixel_point_on_the_left_ref_picked_pixel_point_ = nullptr;
        }

        if(ref_u + 1 < width)
        {
            ptr_pixel_point_on_the_right_ref_picked_pixel_point_ = ref_ptr_pixel_point_matrix + int(ref_v)* width + int(ref_u+1);

        }
        else{
            ptr_pixel_point_on_the_right_ref_picked_pixel_point_ = nullptr;
        }

        if(ref_v - 1 >= 0){
            
            ptr_pixel_point_on_the_down_ref_picked_pixel_point_ = ref_ptr_pixel_point_matrix + int(ref_v-1)* width + int(ref_u);

        }
        else{
            ptr_pixel_point_on_the_down_ref_picked_pixel_point_ = nullptr;
        }

        if(ref_v + 1 < height){

            ptr_pixel_point_on_the_up_ref_picked_pixel_point_ = ref_ptr_pixel_point_matrix + int(ref_v+1)* width + int(ref_u);

        }
        else{
            ptr_pixel_point_on_the_up_ref_picked_pixel_point_ = nullptr;
        }

        const cv::Mat& ref_cv_bgr_data = ref_image->getBGRData();
        const cv::Mat& tar_cv_bgr_data = tar_image->getBGRData();

        cv::Mat vis_cv_ref_rgb_data = ref_cv_bgr_data.clone();
        cv::Mat vis_cv_tar_rgb_data = tar_cv_bgr_data.clone();


        DebugInfo& debug_info = ptr_picked_ref_ptr_pixel_point_->debug_info_;
        // std::cout << "debug_info.costs " << debug_info.costs_.size() << std::endl;
        if(ptr_picked_ref_ptr_pixel_point_->costs_.size() > 0)
        {
      
            cv::Point2i cv_uv_manual(round(ptr_picked_ref_ptr_pixel_point_->valid_uvs_[debug_info.manual_step_idx_][0]),round(ptr_picked_ref_ptr_pixel_point_->valid_uvs_[debug_info.manual_step_idx_][1]));
            if(ptr_picked_ref_ptr_pixel_point_->status_ == PixelPoint::Status::NORMAL || ptr_picked_ref_ptr_pixel_point_->status_ == PixelPoint::Status::INVALID_NON_UNIQUENESSS){

                cv::Point2i cv_uv_best(round(debug_info.uv_possible_match_[0]),round(debug_info.uv_possible_match_[1]));
                cv::circle(vis_cv_tar_rgb_data, cv_uv_best, 2, cv::Scalar(255,0,255), 1); // pink
            }

            cv::Point2i cv_uv_start(round(ptr_picked_ref_ptr_pixel_point_->valid_uvs_.front()[0]),round(ptr_picked_ref_ptr_pixel_point_->valid_uvs_.front()[1]));
            cv::Point2i cv_uv_end(round(ptr_picked_ref_ptr_pixel_point_->valid_uvs_.back()[0]),round(ptr_picked_ref_ptr_pixel_point_->valid_uvs_.back()[1]));

            cv::circle(vis_cv_tar_rgb_data, cv_uv_manual, 2, cv::Scalar(0,255,0), 1); // green
            cv::circle(vis_cv_tar_rgb_data, cv_uv_start, 2, cv::Scalar(0,0,255), 1); // red
            cv::putText(vis_cv_tar_rgb_data, "S", cv_uv_start, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,255), 1.8);
            cv::line(vis_cv_tar_rgb_data, cv_uv_start, cv_uv_end, cv::Scalar(0,255,0), 1); // green
            cv::circle(vis_cv_tar_rgb_data, cv_uv_end, 2, cv::Scalar(255,0,0), 1); // blue
            cv::putText(vis_cv_tar_rgb_data, "E", cv_uv_end, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,0,0), 1.8);
        
        }

        
        cv::circle(vis_cv_ref_rgb_data, cv::Point2i(ref_u,ref_v), 2, cv::Scalar(0,0,255), 1);
        cv::putText(vis_cv_ref_rgb_data, "("+std::to_string(ref_u)+","+std::to_string(ref_v)+")",cv::Point2i(ref_u,ref_v), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,0,255), 1.8);


        std::shared_ptr<glk::Texture> ref_image_texture = glk::create_texture(vis_cv_ref_rgb_data);
        std::shared_ptr<glk::Texture> tar_image_texture = glk::create_texture(vis_cv_tar_rgb_data);

        viewer_->update_image("images/ref_image", ref_image_texture, 1.0);
        viewer_->update_image("images/tar_image", tar_image_texture, 1.0);
    }
}


void IridescenceVisualizer::showInterface()
{


    showReconstruction();

    while (viewer_->spin_once()) {
        
        // std::cout << "inside loop" << std::endl;

        if(isNextAction_  == true)
        {
            isNextAction_  = false;
            break;
        }
    }

    // std::cout << "test" << std::endl;

}


void IridescenceVisualizer::showRefImagePointCloud(Image* ref_image)
{

    // bool isSuccess = false;
    // isSuccess = ref_image->loadData();
    // if(isSuccess == false){
    //     return;
    // }

    PixelPoint* ref_ptr_pixel_point_matrix = ref_image->getPixelPointMatrixPtr();


    std::vector<Image*>& images = dataset_->getImages();
    int height = ref_image->getHeight();
    int width = ref_image->getWidth();
    Eigen::Matrix3f K_cam0 = config_->cameras_[0].getIntrinsicsMatrix();
    Eigen::Matrix4f T_first_ref_image_world = MVS::invertTransform(images[0]->getTransformationMatrix());
    Eigen::Matrix4f T_world_ref_image = T_first_ref_image_world * ref_image->getTransformationMatrix();
    // Eigen::Matrix4f T_world_ref_image = ref_image->getTransformationMatrix();


    std::vector<Eigen::Vector3f> point_cloud_vertices;
    std::vector<Eigen::Vector4f> point_cloud_colors;

    for(int v=0;v<height;v++)
    {
        for(int u=0;u<width;u++)
        {   
            int coord = v*width+u;
            PixelPoint& ref_ptr_pixel_point = ref_ptr_pixel_point_matrix[coord];
    
            float min_z = ref_ptr_pixel_point.min_depth_;
            float max_z = ref_ptr_pixel_point.max_depth_;
            float z = ref_ptr_pixel_point.output_data_.depth_;

            if(max_z - min_z > 0.5f || z < config_->min_depth_ || z > config_->max_depth_)
            {
                continue;
            }

            MultiViewStereo::Status multi_view_stereo_status = multi_view_stereo_->getStatus();
            if(isShowOnConsistencycheck_ == true && multi_view_stereo_status == MultiViewStereo::Status::FREE)
            {
                if(ref_ptr_pixel_point.num_correct_consistency_check_ < 2)
                {
                    continue;
                }

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

   
    // viewer_->update_drawable("points", point_cloud_buffer_ptr, *point_cloud_buffer_shader_setting);
    
    std::string point_label = "point-" + std::to_string(ref_image->getId());
    viewer_->update_drawable(point_label, point_cloud_buffer_ptr, *point_cloud_buffer_shader_setting);

}

void IridescenceVisualizer::showReconstruction()
{
    
    Image* const ref_image = multi_view_stereo_->getReferenceImage();
    Image* const tar_image = multi_view_stereo_->getTargetImage();

    bool isSuccess = false;
    isSuccess = ref_image->loadData();
    if(isSuccess == false){
        return;
    }

    isSuccess = tar_image->loadData();
    if(isSuccess == false){
        return;
    }

    std::vector<Image*>& images = dataset_->getImages();
    int height = ref_image->getHeight();
    int width = ref_image->getWidth();
    Eigen::Matrix3f K_cam0 = config_->cameras_[0].getIntrinsicsMatrix();
    Eigen::Matrix4f T_first_ref_image_world = MVS::invertTransform(images[0]->getTransformationMatrix());
    Eigen::Matrix4f T_world_ref_image = T_first_ref_image_world * ref_image->getTransformationMatrix();
    // Eigen::Matrix4f T_world_ref_image = ref_image->getTransformationMatrix();

    const cv::Mat& ref_cv_bgr_data = ref_image->getBGRData();
    const cv::Mat& tar_cv_bgr_data = tar_image->getBGRData();

    PixelPoint* ref_ptr_pixel_point_matrix = ref_image->getPixelPointMatrixPtr();

    cv::Mat ref_colorized_depth(height, width, CV_8UC3);
    ColorizedCVDepth(config_, ref_ptr_pixel_point_matrix, width, height, config_->min_depth_, 40, ref_colorized_depth);

    std::shared_ptr<glk::Texture> ref_image_texture = glk::create_texture(ref_cv_bgr_data);
    std::shared_ptr<glk::Texture> ref_colorized_depth_texture = glk::create_texture(ref_colorized_depth);
    std::shared_ptr<glk::Texture> tar_image_texture = glk::create_texture(tar_cv_bgr_data);
    ref_image_canvas_->set_texture(ref_image_texture);

    viewer_->update_image("images/ref_image", ref_image_texture, 1.0);
    viewer_->update_image("images/ref_colorized_depth_image", ref_colorized_depth_texture, 0.5);
    viewer_->update_image("images/tar_image", tar_image_texture, 1.0);


    depth_uncertainty_viewer_->update_drawable("ref_image_canvas_", ref_image_canvas_, guik::TextureColor());


    for(size_t i=0;i < images.size(); i++)
    {   
        MVS::Image* image = images[i];
        int image_pose_id =  image->getPoseId();
        int cam_id = image->getCameraId();

        Eigen::Matrix4f T_world_image = Tcl_ * T_first_ref_image_world * image->getTransformationMatrix();
        // Eigen::Matrix4f T_world_image = Tcl_ * image->getTransformationMatrix();

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

    std::vector<Eigen::Vector3f> depth_uncertainty_vertices;
    for(int v=0;v<height;v++)
    {
        for(int u=0;u<width;u++)
        {   
            int coord = v*width+u;
            PixelPoint& ref_ptr_pixel_point = ref_ptr_pixel_point_matrix[coord];


            float min_z = ref_ptr_pixel_point.min_depth_;
            float max_z = ref_ptr_pixel_point.max_depth_;
            float z = ref_ptr_pixel_point.output_data_.depth_;
            float diff_z = max_z - min_z;
            if(diff_z > 100)
            {
                diff_z = 100;
            }

            depth_uncertainty_vertices.push_back(Eigen::Vector3f(u,v,0.0f));
            depth_uncertainty_vertices.push_back(Eigen::Vector3f(u,v,diff_z));
        }
    }

    depth_uncertainty_viewer_->update_drawable("depth uncertainty grid", std::make_shared<glk::ThinLines>(depth_uncertainty_vertices), guik::FlatColor(0.0f, 1.0f, 0.0f, 0.5f));


 
    if(selected_idx_display_mode_ == 0){
        showRefImagePointCloud(ref_image);
    }
    else if(selected_idx_display_mode_ == 1)
    {
        for(int idx = 0; idx < images.size(); idx++)
        {
            showRefImagePointCloud(images[idx]);

        }
    }

   
    DebugPlot();


}


} // namespace MVS


