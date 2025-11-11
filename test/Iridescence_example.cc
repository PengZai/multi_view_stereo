#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>
#include <glk/thin_lines.hpp>
#include <glk/colormap.hpp>
#include <glk/texture_opencv.hpp>
#include <glk/mesh.hpp>
#include <glk/texture.hpp>
#include <glk/mesh_model.hpp>
#include <glk/gridmap.hpp>


int main(int argc, char** argv) {
  // Create a viewer instance (global singleton)
  // auto viewer = guik::LightViewer::instance(Eigen::Vector2i(-1, -1), false, "viewer1");


  auto viewer = guik::viewer(Eigen::Vector2i(-1, -1), false, "viewer0");
  // auto viewer2 = guik::viewer(Eigen::Vector2i(-1, -1), false, "viewer2");

  auto sub_viewer1 = viewer->sub_viewer("sub1", Eigen::Vector2i(960, 600));

  float angle = 0.0f;

  int idx_display_view = 0;



  // Register a callback for UI rendering
  viewer->register_ui_callback("ui", [&]() {
    // In the callback, you can call ImGui commands to create your UI.
    // Here, we use "DragFloat" and "Button" to create a simple UI.


    if (ImGui::BeginMenu("File"))
    {
        if (ImGui::MenuItem("View0"))   { idx_display_view = 0; }
        if (ImGui::MenuItem("View1"))   { idx_display_view = 1; }
        ImGui::EndMenu();
    }

    if (ImGui::Button("Close")) {
      viewer->close();
    }

    if (ImGui::Button("Close2")) {
      viewer->close();
    }



    // if (ImGui::BeginMenuBar())
    // {
    //     if (ImGui::BeginMenu("File"))
    //     {
    //         if (ImGui::MenuItem("Open..", "Ctrl+O")) { /* Do stuff */ }
    //         if (ImGui::MenuItem("Save", "Ctrl+S"))   { /* Do stuff */ }
    //         if (ImGui::MenuItem("Close", "Ctrl+W"))  { viewer->close(); }
    //         ImGui::EndMenu();
    //     }
    //     ImGui::EndMenuBar();
    // }

    // auto& io = ImGui::GetIO();
    // if (!io.WantCaptureMouse && io.MouseClicked[ImGuiMouseButton_Right]) {
    //   // Pick the depth of the clicked pixel
    //   float depth = viewer->pick_depth({io.MousePos.x, io.MousePos.y});

    //   if(depth < 1.0f) {
    //     // Compute the 3D position of the clicked pixel
    //     Eigen::Vector3f pos = viewer->unproject({io.MousePos.x, io.MousePos.y}, depth);
    //     viewer->update_drawable("sphere", glk::Primitives::sphere(), guik::FlatRed().translate(pos).scale(0.05));
    //   }
    // }
  });

  cv::Mat image = cv::imread("../figs/vis_ref_rgb_data.png");
  auto texture = glk::create_texture(image);
  auto shader_setting = guik::TextureColor(Eigen::Matrix4f::Identity());

  std::vector<Eigen::Vector3f> vertices;

  for(int v=0;v<480;v++)
  {
    for(int u=0;u<640;u++)
    {
      vertices.push_back(Eigen::Vector3f(u,v,1.0f));
      vertices.push_back(Eigen::Vector3f(u,v,10.0f));

      vertices.push_back(Eigen::Vector3f(u,v,15.0f));
      vertices.push_back(Eigen::Vector3f(u,v,20.0f));

      vertices.push_back(Eigen::Vector3f(u,v,25.0f));
      vertices.push_back(Eigen::Vector3f(u,v,30.0f));



      // viewer->update_wire_frustum("frustum", guik::FlatGreen());
    }
  }
  Eigen::Affine3f Tlc = Eigen::Affine3f::Identity();
  Tlc.linear() <<
     1.0, 0.0,  0,
     0,  0,  1.0,
     0.0,  -1.0,  0.0;
  Tlc.translation() << 0.0f, 5.0f, 0.0; // 5cm right, 10cm up
  // Spin the viewer until it gets closed
  while (viewer->spin_once()) {
    // Objects to be rendered are called "drawables" and managed with unique names.
    // Here, solid and wire spheres are registered to the viewer respectively with the "Rainbow" and "FlatColor" coloring schemes.
    // The "Rainbow" coloring scheme encodes the height of each fragment using the turbo colormap by default.
    // Eigen::AngleAxisf transform(angle, Eigen::Vector3f::UnitZ());
    // viewer->update_drawable("sphere", glk::Primitives::sphere(), guik::Rainbow(transform));
    // viewer->update_drawable("wire_sphere", glk::Primitives::wire_sphere(), guik::FlatColor({0.1f, 0.7f, 1.0f, 1.0f}, transform));
    // std::vector<Eigen::Vector3f> vertices = {
    //         {0.0f, 0.0f, 1.0f},
    //         {0.0f, 0.0f, 2.0f},
    //         {1.0f, 0.0f, 1.0f},
    //         {1.0f, 0.0f, 2.0f},
    //         {0.0f, 1.0f, 1.0f},
    //         {0.0f, 1.0f, 2.0f}

    //     };

    


    auto lines = std::make_shared<glk::ThinLines>(vertices);

    // --------------------------------------------------
    // Set line width
    // --------------------------------------------------
    lines->set_line_width(2.0f);


    sub_viewer1->update_coord("coord", guik::VertexColor());
    sub_viewer1->update_cube("cube", shader_setting);


    viewer->update_drawable("coord_camera", glk::Primitives::coordinate_system(), guik::VertexColor(Tlc));



    viewer->update_drawable("lines", lines, guik::FlatColor(0.0f, 1.0f, 0.0f, 1.0f));
    viewer->update_coord("coord", guik::VertexColor());
    viewer->update_image("image", texture);

    auto model = std::make_shared<glk::MeshModel>();
    model->override_material(guik::TextureColor(), texture);
    viewer->update_drawable("model", model, guik::Rainbow());

    
  }

  return 0;
}