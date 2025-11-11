#include <glk/mesh.hpp>
#include <guik/viewer/light_viewer.hpp>
#include <Eigen/Dense>
#include <glk/texture_opencv.hpp>
#include <guik/viewer/shader_setting.hpp>
#include <glk/primitives/primitives.hpp>
#include <glk/thin_lines.hpp>
#include <glk/drawable.hpp>


#include "../src/iridescence_visualizer.h"

// class CustomCoordSystem : public glk::Drawable{

// }






void make_coord_system_drawables(std::shared_ptr<glk::ThinLines>& coords, float s = 1.0f, float alpha = 1.0f)
{
    
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> coord_vertices = 
    {
            Eigen::Vector3f::Zero(),
            Eigen::Vector3f::UnitX() * s,
            Eigen::Vector3f::Zero(),
            Eigen::Vector3f::UnitY() * s,
            Eigen::Vector3f::Zero(),
            Eigen::Vector3f::UnitZ() * s
    };

    std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>> coord_colors = 
    {
        Eigen::Vector4f(1.0f, 0.0f, 0.0f, alpha),
        Eigen::Vector4f(1.0f, 0.0f, 0.0f, alpha),
        Eigen::Vector4f(0.0f, 1.0f, 0.0f, alpha),
        Eigen::Vector4f(0.0f, 1.0f, 0.0f, alpha),
        Eigen::Vector4f(0.0f, 0.0f, 1.0f, alpha),
        Eigen::Vector4f(0.0f, 0.0f, 1.0f, alpha)
    };


    coords = std::make_shared<glk::ThinLines>(coord_vertices, coord_colors);

}

void make_camera_drawables(std::shared_ptr<glk::Mesh>& sides_wire,
                           std::shared_ptr<glk::Mesh>& front_tex,
                           float w, float h, float z,
                           const std::shared_ptr<glk::Texture>& texture)
{
    // ----- shared geometry (5 vertices: 4 corners + apex) -----
    // std::vector<Eigen::Vector3f> vertices = {
    //     { w/2.f,  h/2.f,  z},  // 0
    //     {-w/2.f,  h/2.f,  z},  // 1
    //     {-w/2.f, -h/2.f,  z},  // 2
    //     { w/2.f, -h/2.f,  z},  // 3
    //     { 0.f,    0.f,    0.f} // 4 (apex)
    // };

    std::vector<Eigen::Vector3f> vertices = {
        {w / 2, -h / 2, z},
        { -w / 2, -h / 2, z},
        { -w / 2, h / 2, z},
        { w / 2, h / 2, z},
        {0,0,0}
    };



    std::vector<unsigned int> side_indices = {
        0,4,1,  1,4,2,  2,4,3,  3,4,0
    };

    // ----- sides: wireframe, no texcoords -----
    sides_wire = std::make_shared<glk::Mesh>(
        vertices.data(),             sizeof(float)*3,   // positions
        nullptr,                     0,                 // normals (none)
        nullptr,                     0,                 // colors (none)
        nullptr,                     0,                 // texcoords (none)
        static_cast<int>(vertices.size()),
        side_indices.data(),
        static_cast<int>(side_indices.size()),
        /*wireframe=*/true
    );

    // ----- front: its own 4-vertex list + texcoords + texture -----
    std::vector<Eigen::Vector3f> v_front = {
        vertices[0], 
        vertices[1], 
        vertices[2], 
        vertices[3]
    };
    std::vector<unsigned int> front_indices = { 
        0,2,1,
        0,3,2 
    };

    std::vector<Eigen::Vector4f> front_vertice_colors = {
      {1.0f, 1.0f, 1.0f, 1.0f},
      {1.0f, 1.0f, 1.0f, 1.0f},
      {1.0f, 1.0f, 1.0f, 1.0f},
      {1.0f, 1.0f, 1.0f, 1.0f}
    };


    // UVs for the front face (flip V if your image is upside-down)
    std::vector<Eigen::Vector2f> uv_front = {
        {1.f, 0.f},  // for v_front[0]
        {0.f, 0.f},  // for v_front[1]
        {0.f, 1.f},  // for v_front[2]
        {1.f, 1.f}   // for v_front[3]
        // {0.0f, 0.0f},
        // {1.0f, 0.0f},
        // {1.0f, 1.0f},
        // {0.0f, 1.0f}
    };

    std::vector<Eigen::Vector3f> normals = {
        {0.f,0.f,1.f}
    };


    front_tex = std::make_shared<glk::Mesh>(
        v_front.data(),              sizeof(float)*3,
        nullptr,    0,              // normals (optional)
        front_vertice_colors.data(),                     sizeof(float)*4,                 // colors (none)
        uv_front.data(),             sizeof(float)*2,   // texcoords
        static_cast<int>(v_front.size()),
        front_indices.data(),
        static_cast<int>(front_indices.size()),
        /*wireframe=*/false
    );
    front_tex->set_texture(texture);
}

void make_camera_mesh(std::shared_ptr<glk::Mesh>& mesh, float w, float h, float z, bool close_front, std::shared_ptr<glk::Texture> texture)
{

  std::vector<Eigen::Vector3f> vertices = {
    {w / 2, h / 2, z},
    { -w / 2, h / 2, z},
    { -w / 2, -h / 2, z},
    { w / 2, -h / 2, z},
    {0,0,0}
  };

//   std::vector<Eigen::Vector3f> vertices = {
//     {w / 2, -h / 2, z},
//     { -w / 2, -h / 2, z},
//     { -w / 2, h / 2, z},
//     { w / 2, h / 2, z},
//     {0,0,0}
//   };

    std::vector<Eigen::Vector3f> normals;


    normals.push_back(vertices[0].normalized());    // n1
    normals.push_back(vertices[1].normalized());    // n2
    normals.push_back(vertices[2].normalized());    // n3
    normals.push_back(vertices[3].normalized());    // n4
    normals.push_back(Eigen::Vector3f(0, 0, 0));    // n5


    std::vector<unsigned int> indices = {
        0, 4, 1, 
        1, 4, 2, 
        2, 4, 3, 
        3, 4, 0
    };

    // if (close_front) {
    //   std::array<int, 6> front_indices = { 0, 1, 2, 0, 2, 3};
    //   indices.insert(indices.end(), front_indices.begin(), front_indices.end());
    // }

    std::vector<Eigen::Vector2f> tex_coords = {
        {1.f, 0.f}, // match your preferred orientation
        {0.f, 0.f},
        {0.f, 1.f},
        {1.f, 1.f}
    };




    mesh = std::make_shared<glk::Mesh>(
        vertices.data(), sizeof(float)*3,
        normals.data(), sizeof(float) * 3,
        nullptr, 0,   // no colors
        tex_coords.data(), sizeof(float) * 2,
        vertices.size(),
        indices.data(),
        indices.size(),
        true
    );

    mesh->set_texture(texture);

}

int main() {
  guik::LightViewer* viewer = guik::LightViewer::instance();
  

  // Define vertices for a square (on XY plane, z = 0)
  std::vector<Eigen::Vector3f> vertices = {
    {0.0f, 0.0f, 0.0f},
    { 640.0f, 0.0f, 0.0f},
    { 640.0f, 480.0f, 0.0f},
    { 0.0f,  480.0f, 0.0f}
  };

  std::vector<unsigned int> indices = {
    0, 1, 2,   // triangle 1
    0, 2, 3    // triangle 2
  };

  std::vector<Eigen::Vector2f> tex_coords = {
    {0.0f, 0.0f},
    {1.0f, 0.0f},
    {1.0f, 1.0f},
    {0.0f, 1.0f}
  };


  auto picture_mesh = std::make_shared<glk::Mesh>(
    vertices.data(), sizeof(float)*3,
    nullptr, 0,
    nullptr, 0,   // no colors
    tex_coords.data(), sizeof(float) * 2,
    vertices.size(),
    indices.data(),
    indices.size()
  );
  
  cv::Mat image = cv::imread("../figs/vis_ref_rgb_data.png");
  auto texture = glk::create_texture(image);
  picture_mesh->set_texture(texture);

  int front_color_mode = guik::ColorMode::TEXTURE_COLOR;
  int side_color_mode = guik::ColorMode::FLAT_COLOR;

  Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
  
  transformation <<
     1.0f, 0.0f, 0.0f, 0.0f,
     0.0f, 0.0f, 1.0f, 5.0f,
     0.0f,-1.0f, 0.0f, 2.0f,
     0.0f, 0.0f, 0.0f, 1.0f;


  auto ground_shader_setting = guik::ShaderSetting(guik::ColorMode::TEXTURE_COLOR, Eigen::Matrix4f::Identity());

  auto camera_coord_shader_setting = guik::ShaderSetting(guik::ColorMode::FLAT_COLOR, transformation);

  auto front_shader_setting = guik::ShaderSetting(guik::ColorMode::TEXTURE_COLOR, transformation);
  // front_shader_setting.set_color(Eigen::Vector4f({0.f, 1.f, 0.f, 0.5f}));
  // front_shader_setting.make_transparent();
  front_shader_setting.set_alpha(0.5f);

  auto side_shader_setting = guik::ShaderSetting(guik::ColorMode::FLAT_COLOR, transformation);
  side_shader_setting.set_color(Eigen::Vector4f({0.f, 1.f, 0.f, 0.5f}));

  auto wire_frustum = glk::Primitives::wire_frustum();
  std::shared_ptr<glk::Mesh> mesh;
  make_camera_mesh(mesh , 0.6f, 0.4f, 0.5f, true, texture);

  std::shared_ptr<glk::Mesh> sides_wire, front_tex;
  make_camera_drawables(sides_wire, front_tex, 1.0f, 0.6f, 0.8f, texture);
  
  std::shared_ptr<glk::ThinLines> coords;
  make_coord_system_drawables(coords, 1.0f, 1.0f);


  Eigen::Matrix4f transformation2 = Eigen::Matrix4f::Identity();
  transformation2 <<
     1.0f, 0.0f, 0.0f, 0.0f,
     0.0f, 0.0f, 1.0f, 10.0f,
     0.0f,-1.0f, 0.0f, 2.0f,
     0.0f, 0.0f, 0.0f, 1.0f;

  Eigen::Matrix4f transformation3 = Eigen::Matrix4f::Identity();
  transformation3 <<
     1.0f, 0.0f, 0.0f, 0.0f,
     0.0f, 0.0f, 1.0f, 15.0f,
     0.0f,-1.0f, 0.0f, 2.0f,
     0.0f, 0.0f, 0.0f, 1.0f;


  // auto camera_model_shader_setting = guik::ShaderSetting(guik::ColorMode::VERTEX_COLOR, transformation2);
  // camera_model_shader_setting.set_alpha(0.4f);
  // camera_model_shader_setting.make_transparent();
  std::shared_ptr<MVS::CameraModel> camera_model = std::make_shared<MVS::CameraModel>(2.0f, Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f));
    std::shared_ptr<MVS::CameraModel> camera_model3 = std::make_shared<MVS::CameraModel>(2.0f, Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f));

  auto coord_camera = glk::Primitives::coordinate_system();

  camera_model->setTransform(transformation2);



  while (viewer->spin_once()) {
 

 
    viewer->update_drawable("picture_mesh", picture_mesh, ground_shader_setting);
    viewer->update_coord("coord", guik::VertexColor());
    // viewer->update_drawable("wire_frustum", mesh, shader_setting);
    viewer->update_drawable("front", front_tex, front_shader_setting);
    viewer->update_drawable("sides", sides_wire, side_shader_setting);
    viewer->update_drawable("coord_camera", coords, guik::VertexColor().transform(transformation).make_transparent());

    camera_model->setTransform(transformation2);
    camera_model->setColor(Eigen::Vector4f(1.0f, 0.0f, 0.0f, 0.5f));
    camera_model->draw(viewer, "0");
    camera_model3->setTransform(transformation3);
    camera_model3->setColor(Eigen::Vector4f(0.0f, 1.0f, 0.0f, 0.5f));
    camera_model3->draw(viewer, "1");



  }

  return 0;
}
