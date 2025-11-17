#include "multi_view_stereo.h"

namespace MVS
{


MultiViewStereo::MultiViewStereo(Config* const config)
:ref_image_(nullptr)
{
    config_ = config;
    half_ws_ = config_->half_window_size_;
    // max_inv_depth_ = 1/config->min_depth_;
    // min_inv_depth_ = 1/config->max_depth_;


}

MultiViewStereo::~MultiViewStereo()
{


}

void MultiViewStereo::setDataset(Dataset* const dataset)
{
    dataset_ = dataset;
}

void MultiViewStereo::setVisualizer(Visualizer* const visualizer)
{
    visualizer_ = visualizer;
}


void MultiViewStereo::setReferenceImage(Image* const image)
{
    ref_image_ = image;
}
 


void MultiViewStereo::run()
{   

    std::vector<Image*> images = dataset_->getImages();

    for(size_t ref_image_idx=0; ref_image_idx < images.size(); ref_image_idx++){
        Image* ref_image = images[ref_image_idx];
        int ref_pose_id = ref_image->getPoseId();
        int ref_camera_id = ref_image->getCameraId();

        if(ref_pose_id == config_->ref_pose_idx_ && ref_camera_id == config_->ref_camera_idx_)
        {
            setReferenceImage(ref_image);
            break;
        }
    }

    if(ref_image_ == nullptr){
        std::cout << "ref_image is not existed for ref_pose_id:" << config_->ref_pose_idx_ << " and ref_camera_id:" << config_->ref_camera_idx_ << std::endl;
        return;
    }
    ref_image_->loadData();


    Image* tar_image = nullptr;
    for(size_t tar_image_idx=0; tar_image_idx < images.size(); tar_image_idx++)
    {
        tar_image = images[tar_image_idx];
        bool isSuccess = tar_image->loadData();        
        if(isSuccess == false){
            continue;
        }
        int tar_pose_id = tar_image->getPoseId();
        int tar_camera_id = tar_image->getCameraId();

        if(ref_image_->getId() == tar_image->getId()){
            continue;
        }

        if(tar_pose_id < config_->tar_pose_start_idx_ || tar_camera_id < config_->tar_camera_start_idx_)
        {
            continue;
        }

        printf("match start for ref pose %d, cam %d and tar pose %d, cam %d\n", ref_image_->getPoseId(), ref_image_->getCameraId(), tar_image->getPoseId(), tar_image->getCameraId());

        auto start = std::chrono::steady_clock::now();        
        match(ref_image_, tar_image, config_->debug_plot_);


        auto end = std::chrono::steady_clock::now();
        auto tt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("match Done! Timing : %lf s for ref pose %d, cam %d and tar pose %d, cam %d\n", tt.count() / 1000.0, ref_image_->getPoseId(), ref_image_->getCameraId(), tar_image->getPoseId(), tar_image->getCameraId());

        ref_image_->loadDepthFromMinMaxInvDepth();

        // visualizer_->saveDepth(ref_image_);

        visualizer_->showRefImageReconstruction(ref_image_, tar_image);
        

    }

    
    // visualizer_->showUndistortedGrayImage(ref_image_, "RefUndistortedGray");
    // visualizer_->showUndistortedGrayImage(tar_image, "TarUndistortedGray");

    visualizer_->saveDepth(ref_image_);

    // cv::waitKey(0);

}

void MultiViewStereo::match(Image* const ref_image, Image* const tar_image, bool debug_plot)
{

    Eigen::Matrix3f K_ref = config_->cameras_[ref_image->getCameraId()].getIntrinsicsMatrix();
    Eigen::Matrix3f K_tar = config_->cameras_[tar_image->getCameraId()].getIntrinsicsMatrix();
    Eigen::Matrix4f T_world_ref = ref_image->getTransformationMatrix();
    Eigen::Matrix4f T_world_tar = tar_image->getTransformationMatrix();
    Eigen::Matrix4f T_tar_ref =  invertTransform(T_world_tar) * T_world_ref;


    Eigen::Matrix3f R_tar_ref = T_tar_ref.block<3,3>(0,0);
    Eigen::Vector3f t_tar_ref = T_tar_ref.block<3,1>(0,3);
    
    Eigen::Matrix3f KRKi = K_tar * R_tar_ref * K_ref.inverse();
    Eigen::Vector3f Kt = K_tar * t_tar_ref;

    const cv::Mat ref_rgb_data = ref_image->getBGRData();

    PixelPoint* ptr_pixel_point_matrix = ref_image->getPixelPointMatrixPtr();


    int width = ref_image->getWidth();
    int height = ref_image->getHeight();

    for(size_t v=half_ws_; v < height - half_ws_; v++)
    {
        for(size_t u=half_ws_; u < width - half_ws_; u++)
        {
            // KRKi * (u,v,1) + dmin_inv * Kt = pt_min
            // KRKi * (u,v,1) + dmax_inv * Kt = pt_max
            // u_min, v_min = pt_min / pt_min[2]
            // u_max, v_max = pt_max / pt_max[2]

            if(v < config_->start_match_uv_[1] || u < config_->start_match_uv_[0]){
                continue;
            }

            // std::cout << "epipolar search for " << u << ", "<< v << std::endl;

            // if(debug_plot)
            // {
            //     cv::Mat vis_ref_rgb_data = ref_rgb_data.clone();
            //     cv::circle(vis_ref_rgb_data, cv::Point2i(u,v), 3, cv::Scalar(0,0,255), 2);
            //     cv::putText(vis_ref_rgb_data, "("+std::to_string(u)+","+std::to_string(v)+")",cv::Point2i(u,v), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,0,255), 1.8);

            //     cv::imwrite(config_->save_figure_path_ + "/vis_ref_rgb_data.png", vis_ref_rgb_data);
            //     cv::imshow("vis_ref_rgb_data", vis_ref_rgb_data);

            // }
        
            // auto start = std::chrono::steady_clock::now();        
            int current_coord = v*width + u;
            Eigen::Vector3f KRKi_uv_homo = KRKi * Eigen::Vector3f(u,v,1);
            // Eigen::Vector3f pt_min = KRKi_uv_homo + (1.0/config_->max_depth_) * Kt;
            // Eigen::Vector3f pt_max = KRKi_uv_homo + (1.0/config_->min_depth_) * Kt;
            PixelPoint& pixel_point = ptr_pixel_point_matrix[current_coord];
            for(int min_max_idx=0; min_max_idx<(int)pixel_point.min_inv_depth_vec_.size();min_max_idx++)
            {

                Eigen::Vector3f pt_min = KRKi_uv_homo + pixel_point.min_inv_depth_vec_[min_max_idx] * Kt;
                Eigen::Vector3f pt_max = KRKi_uv_homo + pixel_point.max_inv_depth_vec_[min_max_idx] * Kt;
                Eigen::Vector2f uv_min = Eigen::Vector2f(pt_min[0]/pt_min[2], pt_min[1]/pt_min[2]);
                Eigen::Vector2f uv_max = Eigen::Vector2f(pt_max[0]/pt_max[2], pt_max[1]/pt_max[2]);

                if(debug_plot == true)
                {

                    pixel_point.debug_info_vec_[min_max_idx].clean();

                    pixel_point.debug_info_vec_[min_max_idx].tar_pose_id_ = tar_image->getPoseId();
                    pixel_point.debug_info_vec_[min_max_idx].tar_camera_id_ = tar_image->getCameraId();
                    pixel_point.debug_info_vec_[min_max_idx].init_min_depth_ = 1/pixel_point.max_inv_depth_vec_[min_max_idx];
                    pixel_point.debug_info_vec_[min_max_idx].init_max_depth_ = 1/pixel_point.min_inv_depth_vec_[min_max_idx];
                    pixel_point.debug_info_vec_[min_max_idx].init_depth_ = (pixel_point.debug_info_vec_[min_max_idx].init_min_depth_ + pixel_point.debug_info_vec_[min_max_idx].init_max_depth_) / 2.0;
                    pixel_point.debug_info_vec_[min_max_idx].uv_min_ = uv_min;
                    pixel_point.debug_info_vec_[min_max_idx].uv_max_ = uv_max;

                    
                }

                
                Eigen::Vector2f uv_best_match;
                // auto end = std::chrono::steady_clock::now();    
                // auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                // printf("epipolar line generation ! Timing : %lld µs\n", (long long)elapsed_us);

                // start = std::chrono::steady_clock::now();        
                // const cv::Mat ref_gray_patch = getSubpixelPatch(ref_gray_data, u,  v, ws_, ws_);
                // uint8_t* ref_ptr_gray_patch = ref_ptr_gray_data + v*width + u;
                // getRoundPixelPatch(ref_gray_data, u, v, half_ws_, half_ws_, ref_gray_patch);

                // end = std::chrono::steady_clock::now();
                // elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                // printf("getSubpixelPatch done ! Timing : %lld µs\n", (long long)elapsed_us);

                // auto start = std::chrono::steady_clock::now();        
                bool isValid = epipolarSearch(ptr_pixel_point_matrix, u, v, min_max_idx, tar_image, uv_min, uv_max, uv_best_match, debug_plot);
                // auto end = std::chrono::steady_clock::now();
                // auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                // printf("epipolar search done ! Timing : %lld µs\n", (long long)elapsed_us);

                // dx*dx > dy*dy
                float inv_depth = 0;
                float const_error_in_pixel = 1;
                if(isValid == true){
                    // if(unit_epipolar_vector[0] * unit_epipolar_vector[0] >= unit_epipolar_vector[1] * unit_epipolar_vector[1])
                    // {   
                    //     inv_depth = (KRKi_uv_homo[2]*uv_best_match[0] - KRKi_uv_homo[0])/(Kt[0] - Kt[2]*uv_best_match[0]);

                    // }
                    // else
                    // {
                    //     inv_depth = (KRKi_uv_homo[2]*uv_best_match[1] - KRKi_uv_homo[1])/(Kt[1] - Kt[2]*uv_best_match[1]);

                    // }

                    // A = (a0, a1)^T, a0 = (Kt[0] - Kt[2]*uv_best_match[0]), a1 = (Kt[1] - Kt[2]*uv_best_match[1])
                    // b = (b0, b1)^T, b0 = (KRKi_uv_homo[2]*uv_best_match[0] - KRKi_uv_homo[0]), b1 = (KRKi_uv_homo[2]*uv_best_match[1] - KRKi_uv_homo[1])
                    // least square solution: inv_d = (A^T * A)^(-1) * A^T * b
                    // inv_d = (a0*b0 + a1*b1) / (a0*a0 + a1*a1)
                    // float a0 = (Kt[0] - Kt[2]*(uv_best_match[0]));
                    // float a1 = (Kt[1] - Kt[2]*(uv_best_match[1]));
                    // float b0 = (KRKi_uv_homo[2]*(uv_best_match[0]) - KRKi_uv_homo[0]);
                    // float b1 = (KRKi_uv_homo[2]*(uv_best_match[1]) - KRKi_uv_homo[1]);
                    // float inv_depth = (a0*b0 + a1*b1) / (a0*a0 + a1*a1);
                    // float depth = 1.0/inv_depth;
                    // if(depth >= config_->min_depth_ && depth <= config_->max_depth_)
                    // {
                    //     ref_ptr_depth_data[current_coord] = depth;
                    // }
                    // else{
                    //     ref_ptr_depth_data[current_coord] = 0.0;
                    // }
                    // if(debug_plot){
                    //     std::cout << "depth : " << depth << " at uv_best_match (" << uv_best_match[0] << "," << uv_best_match[1] << ")" << std::endl;
                    // }

                    Eigen::Vector2f epipolar_vector = Eigen::Vector2f(uv_max[0] - uv_min[0], uv_max[1] - uv_min[1]);
                    float epipolar_length = epipolar_vector.norm();
                    Eigen::Vector2f unit_epipolar_vector =  epipolar_vector / epipolar_length;

                    float a0_minus = (Kt[0] - Kt[2]*(uv_best_match[0]-const_error_in_pixel*unit_epipolar_vector[0]));
                    float a1_minus = (Kt[1] - Kt[2]*(uv_best_match[1]-const_error_in_pixel*unit_epipolar_vector[1]));
                    float b0_minus = (KRKi_uv_homo[2]*(uv_best_match[0]-const_error_in_pixel*unit_epipolar_vector[0]) - KRKi_uv_homo[0]);
                    float b1_minus = (KRKi_uv_homo[2]*(uv_best_match[1]-const_error_in_pixel*unit_epipolar_vector[1]) - KRKi_uv_homo[1]);

                    float inv_min_depth = (a0_minus*b0_minus + a1_minus*b1_minus) / (a0_minus*a0_minus + a1_minus*a1_minus);


                    float a0_plus = (Kt[0] - Kt[2]*(uv_best_match[0]+const_error_in_pixel*unit_epipolar_vector[0]));
                    float a1_plus = (Kt[1] - Kt[2]*(uv_best_match[1]+const_error_in_pixel*unit_epipolar_vector[1]));
                    float b0_plus = (KRKi_uv_homo[2]*(uv_best_match[0]+const_error_in_pixel*unit_epipolar_vector[0]) - KRKi_uv_homo[0]);
                    float b1_plus = (KRKi_uv_homo[2]*(uv_best_match[1]+const_error_in_pixel*unit_epipolar_vector[1]) - KRKi_uv_homo[1]);
                    
                    float inv_max_depth = (a0_plus*b0_plus + a1_plus*b1_plus) / (a0_plus*a0_plus + a1_plus*a1_plus);
                    
                    if(inv_max_depth <= 0)
                    {
                        inv_max_depth = 1.0/config_->min_depth_;
                    }
                    if(inv_min_depth <= 0)
                    {
                        inv_min_depth = 1e-8;
                    }
                    

                    if(inv_min_depth > inv_max_depth) std::swap<float>(inv_min_depth, inv_max_depth);

                    ptr_pixel_point_matrix[current_coord].min_inv_depth_vec_[min_max_idx] = inv_min_depth;
                    ptr_pixel_point_matrix[current_coord].max_inv_depth_vec_[min_max_idx] = inv_max_depth;
                    if(debug_plot == true){
                
                        float min_depth = 1.0/inv_max_depth;
                        float max_depth = 1.0/inv_min_depth;
                        
                        pixel_point.debug_info_vec_[min_max_idx].min_depth_ = min_depth;
                        pixel_point.debug_info_vec_[min_max_idx].max_depth_ = max_depth;
                        pixel_point.debug_info_vec_[min_max_idx].depth_ = (min_depth + max_depth)/2.0;

                        for(int i = 0; i < pixel_point.debug_info_vec_[min_max_idx].valid_uvs_.size();i++)
                        {
                            
                            Eigen::Vector2f valid_uv = pixel_point.debug_info_vec_[min_max_idx].valid_uvs_[i];

                            float a0 = Kt[0] - Kt[2]*valid_uv[0];
                            float a1 = Kt[1] - Kt[2]*valid_uv[1];
                            float b0 = KRKi_uv_homo[2]*valid_uv[0] - KRKi_uv_homo[0];
                            float b1 = KRKi_uv_homo[2]*valid_uv[1] - KRKi_uv_homo[1];
                            float inv_depth = (a0*b0 + a1*b1) / (a0*a0 + a1*a1);
                            pixel_point.debug_info_vec_[min_max_idx].valid_inv_depths_.push_back(inv_depth);
                            pixel_point.debug_info_vec_[min_max_idx].valid_depths_.push_back(1.0f/inv_depth);
                        }

                        // std::cout << "min_depth : " << min_depth << " and max_depth : " << max_depth << " at uv_best_match (" << uv_best_match[0] << "," << uv_best_match[1] << ")" << std::endl;

   
                    }
                    
    
                }
                else
                {
                    if(debug_plot == true){
                
                        pixel_point.debug_info_vec_[min_max_idx].clean();
   
                    }
                }


            }
            
            

            // std::cout << "u,v,d: " << u << "," <<  v <<  "," << ref_ptr_depth_data[v*width+u]  << std::endl;
        }

    }

    if(debug_plot)
    {
    }

}


void MultiViewStereo::getValidTarUV(Eigen::Vector2f& valid_tar_uv, const Eigen::Vector2f tar_uv, const std::vector<Eigen::Vector2f>& intersections_in_tar_image)
{

    int min_idx = -1;
    float min_diff = std::numeric_limits<float>::infinity();
    for(int i=0;i<(int)intersections_in_tar_image.size();i++){
        float diff = (tar_uv - intersections_in_tar_image[i]).norm();
        if(diff < min_diff)
        {
            min_diff = diff;
            min_idx = i;
        }
    }

    valid_tar_uv = intersections_in_tar_image[min_idx];

}

 

bool MultiViewStereo::epipolarSearch(PixelPoint* const ref_ptr_pixel_point_matrix, uint32_t ref_u, uint32_t ref_v, uint32_t min_max_idx, const Image* const tar_image, 
     Eigen::Vector2f &tar_uv_min, Eigen::Vector2f &tar_uv_max, Eigen::Vector2f& tar_uv_best_match, bool debug_plot)
{


    // if( ref_u == 90 && ref_v == 217)
    // {
    //     std::cout << "debug" << std::endl;
    // }
    // if(tar_image->isInImage(tar_uv_max[0], tar_uv_max[1], 0) == false)
    // {
    //     return false;

    // }
    uint32_t width = tar_image->getWidth();
    uint32_t height = tar_image->getHeight();  


        
    // Eigen::Vector2f epipolar_vector = Eigen::Vector2f(tar_uv_max[0] - tar_uv_min[0], tar_uv_max[1] - tar_uv_min[1]);
    // Eigen::Vector2f unit_epipolar_vector =  epipolar_vector / epipolar_length;

    // if(!(tar_uv_max[0] >=0 && tar_uv_max[1] >= 0 && tar_uv_max[0] < width - 1 && tar_uv_max[1] < height - 1))
    // {
    //     std::cout << "out of range " << tar_uv_min[0] << "," << tar_uv_min[1]  << " to " << tar_uv_max[0] << "," << tar_uv_max[1] << " with epi unit " << unit_epipolar_vector[0] << "," << unit_epipolar_vector[1] << std::endl;
    //     return false;   
    // }


    Eigen::Vector2f epipolar_vector = Eigen::Vector2f(tar_uv_max[0] - tar_uv_min[0], tar_uv_max[1] - tar_uv_min[1]);
    float epipolar_length = epipolar_vector.norm();
    Eigen::Vector2f unit_epipolar_vector =  epipolar_vector / epipolar_length;

    if(std::abs(tar_uv_max[0] - tar_uv_min[0]) < 1e-3 || epipolar_length < 1e-3)
    {
        return false;
    }



    bool tar_uv_min_isInImage = tar_image->isInImage(tar_uv_min[0], tar_uv_min[1], half_ws_);
    bool tar_uv_max_isInImage = tar_image->isInImage(tar_uv_max[0], tar_uv_max[1], half_ws_);


    float k = (tar_uv_max[1] - tar_uv_min[1])/(tar_uv_max[0] - tar_uv_min[0]);
    float h1 = half_ws_;
    float h2 = height - half_ws_ - 1;
    float w1 = half_ws_;
    float w2 = width - half_ws_ - 1;
    

    std::vector<Eigen::Vector2f> intersections_in_tar_image;

    Eigen::Vector2f intersect_y_h1 = Eigen::Vector2f((h1 + k*tar_uv_max[0] - tar_uv_max[1])/k, h1);
    if(intersect_y_h1[0] >= half_ws_ && intersect_y_h1[0] <= width - half_ws_ - 1)
    {
        intersections_in_tar_image.emplace_back(intersect_y_h1);
    }
    Eigen::Vector2f intersect_y_h2 = Eigen::Vector2f((h2 + k*tar_uv_max[0] - tar_uv_max[1])/k, h2);
    if(intersect_y_h2[0] >= half_ws_ && intersect_y_h2[0] <= width - half_ws_ - 1)
    {
        intersections_in_tar_image.emplace_back(intersect_y_h2);
    }
    Eigen::Vector2f intersect_x_w1 = Eigen::Vector2f(w1, k*w1+tar_uv_max[1] - k*tar_uv_max[0]);
    if(intersect_x_w1[1] >= half_ws_ && intersect_x_w1[1] <= height - half_ws_ - 1)
    {
        intersections_in_tar_image.emplace_back(intersect_x_w1);
    }
    Eigen::Vector2f intersect_x_w2 = Eigen::Vector2f(w2, k*w2 + tar_uv_max[1] - k*tar_uv_max[0]);
    if(intersect_x_w2[1] >= half_ws_ && intersect_x_w2[1] <= height - half_ws_ - 1)
    {
        intersections_in_tar_image.emplace_back(intersect_x_w2);
    }
  

    if(intersections_in_tar_image.size() == 0)
    {
        return false;
    }

    // case one both in image
    if(tar_uv_min_isInImage == true && tar_uv_max_isInImage == true)
    {
        //do nothing
    }
    else if(tar_uv_min_isInImage == true && tar_uv_max_isInImage == false){


        getValidTarUV(tar_uv_max, tar_uv_max, intersections_in_tar_image);


    }
    else if(tar_uv_min_isInImage == false && tar_uv_max_isInImage == true)
    {

        getValidTarUV(tar_uv_min, tar_uv_min, intersections_in_tar_image);

    }
    else if(tar_uv_min_isInImage == false && tar_uv_max_isInImage == false)
    {

        // tar_uv_min, tar_uv_max were projected in same side.
        if(tar_uv_min[0] < w1 && tar_uv_max[0] < w1 || tar_uv_min[0] > w2 && tar_uv_max[0] > w2 || tar_uv_min[1] < h1 && tar_uv_max[1] < h1 || tar_uv_min[1] > h2 && tar_uv_max[1] > h2 )
        {
            return false;
        }

        getValidTarUV(tar_uv_min, tar_uv_min, intersections_in_tar_image);
        getValidTarUV(tar_uv_max, tar_uv_max, intersections_in_tar_image);

    }



    epipolar_vector = Eigen::Vector2f(tar_uv_max[0] - tar_uv_min[0], tar_uv_max[1] - tar_uv_min[1]);
    epipolar_length = epipolar_vector.norm();
    unit_epipolar_vector =  epipolar_vector / epipolar_length;

    PixelPoint* tar_ptr_pixel_point_matrix = tar_image->getPixelPointMatrixPtr();



  

    float s = 0;
    float min_cost = std::numeric_limits<float>::infinity();
    int best_s_idx = 0;
    int best_s_idx_temp = 0;

    Eigen::Vector2f uv_best_tmp(-1, -1);
    float best_s_tmp = 0;
    

    // for debug_plot
    std::vector<float> costs;                   
    std::vector<float> steps;
    std::vector<Eigen::Vector2f> valid_uvs;
    float ws_2 = (2*half_ws_+1)*(2*half_ws_+1);

    while(s <= epipolar_length)
    {   
        Eigen::Vector2f tar_uv_current  = tar_uv_min + s * unit_epipolar_vector;
        bool isInImage = tar_image->isInImage(tar_uv_current[0], tar_uv_current[1], half_ws_);

        if(isInImage){
            float cost = SAD(ref_ptr_pixel_point_matrix, ref_u, ref_v, tar_ptr_pixel_point_matrix, round(tar_uv_current[0]), round(tar_uv_current[1]), width, height, half_ws_)/(ws_2);
            
            if(debug_plot == true)
            {
                valid_uvs.push_back(tar_uv_current);
                costs.push_back(cost);
                steps.push_back(s);
            }

            
            
            if(cost < min_cost)
            {
                min_cost = cost;
                uv_best_tmp = tar_uv_current;
                best_s_tmp = s;

                if(debug_plot == true){
                    best_s_idx_temp = steps.size()-1;
                }
            }
        }
        
    

        s+=1.0f;

    }


    // do subpixel epipolar search using GN optimization

    // L(s) = \sum ||r(s)||^2
    // residual = \sum ||I_{tar}(u_min+s*dx, v_min+s*dx) - I_{ref}(u_ref, v_ref)||^2
    // Jacobian J = \partial J/ \partial s = \partial I_{tar} / \partial u * \partial u / \parital s + \partial I_{tar} / \partial v * \partial v / \parital s, chain rule
    // J = \grad I_{tar}_u * dx (unit_epipolar_vector[0]) + \Grad I_{tar}_v * dy (unit_epipolar_vector[1])
    // Hessian H = J^T J = J^2
    // b = -J^T * r(s)
    // \delta s = b/H, H\delta s = b

    // size_t maximum_gn_step = 1;

    // for(size_t i=0; i<maximum_gn_step; i++)
    // {
    //     float Hessian = 0;
    //     float b = 0;
    //     float delta_s = 0;
    //     for(int wv =-half_ws_; wv <= half_ws_; wv++)
    //     {
    //         for(int wu =-half_ws_; wu <= half_ws_; wu++)
    //         {
    //             Eigen::Vector2f current_uv = uv_best_tmp + Eigen::Vector2f(wv, wu);
    //             float gray, gx, gy;
    //             gray = getBilinearInterpolated(tar_ptr_gray_data, width, height, current_uv[0], current_uv[1]);
    //             getBilinearInterpolatedGradient(tar_ptr_gradient_gray_data_, width, height, current_uv[0], current_uv[1], gx, gy);
    //             float residual = gray - ref_ptr_gray_data[(ref_v + wv)*width + ref_u + wu];
    //             float Jacobi =  gx*unit_epipolar_vector[0] + gy*unit_epipolar_vector[1];
    //             Hessian+= Jacobi*Jacobi;
    //             b+= Jacobi*residual;
    //         }

    //     }

    //     delta_s = -b/Hessian;
    //     std::cout << "ref_u, v " << ref_u << "," << ref_v << " delta_s " << delta_s << std::endl;
    //     delta_s = std::clamp(delta_s, -1.0f, 1.0f);
    //     best_s_tmp += delta_s;
    //     uv_best_tmp = tar_uv_min + best_s_tmp * unit_epipolar_vector;

    // }

    // std::cout << "epipolar_length : " << epipolar_length << std::endl;
    // min_cost < std::numeric_limits<float>::infinity()
    if(min_cost < ACCEPTABLE_MINI_COST){
        tar_uv_best_match = uv_best_tmp;

        if(debug_plot == true)
        {
            best_s_idx = best_s_idx_temp;

            uint32_t ref_coord = ref_v*width + ref_u;

            ref_ptr_pixel_point_matrix[ref_coord].debug_info_vec_[min_max_idx].best_step_idx_ = best_s_idx;
            ref_ptr_pixel_point_matrix[ref_coord].debug_info_vec_[min_max_idx].uv_best_match_ = tar_uv_best_match;
            ref_ptr_pixel_point_matrix[ref_coord].debug_info_vec_[min_max_idx].costs_ = costs;
            ref_ptr_pixel_point_matrix[ref_coord].debug_info_vec_[min_max_idx].steps_ = steps;
            ref_ptr_pixel_point_matrix[ref_coord].debug_info_vec_[min_max_idx].valid_uvs_ = valid_uvs;
            

            // cv::Point2i cv_uv_best(round(uv_best_tmp[0]),round(uv_best_tmp[1]));
            // cv::Point2i cv_uv_start(round(valid_uvs.front()[0]),round(valid_uvs.front()[1]));
            // cv::Point2i cv_uv_end(round(valid_uvs.back()[0]),round(valid_uvs.back()[1]));

            // matplot::plot(steps, costs, "-o");   // "-o" = line with circle markers

            // matplot::xlabel("Step");
            // matplot::ylabel("Cost");
            // matplot::title("best at  " + std::to_string(cv_uv_best.x) + "," + std::to_string(cv_uv_best.y) + 
            // ", from (" + std::to_string(cv_uv_start.x)+","+std::to_string(cv_uv_start.y) + ") to (" + 
            // std::to_string(cv_uv_end.x)+","+std::to_string(cv_uv_end.y)+")");

            // matplot::text(steps[best_s_idx], costs[best_s_idx], "best");
            // matplot::save(config_->save_figure_path_ + "/cost_vs_step.png");

            // const cv::Mat cv_tar_rgb_data = tar_image->getBGRData();
            // cv::Mat vis_cv_tar_rgb_data = cv_tar_rgb_data.clone();


            // std::cout << "uv_best_tmp : " << tar_uv_best_match[0] << " , " << tar_uv_best_match[1] << " starting from (" 
            //     << cv_uv_start.x << "," << cv_uv_start.y  << ") end at (" << cv_uv_end.x << "," << cv_uv_end.y << ")" << std::endl;
            // cv::circle(vis_cv_tar_rgb_data, cv_uv_best, 3, cv::Scalar(255,0,255), 2); // pink
            // cv::circle(vis_cv_tar_rgb_data, cv_uv_start, 2, cv::Scalar(0,0,255), 2); // red
            // cv::putText(vis_cv_tar_rgb_data, "S",cv_uv_start, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 1.8);
            // cv::line(vis_cv_tar_rgb_data, cv_uv_start, cv_uv_end, cv::Scalar(0,255,0), 1); // green
            // cv::circle(vis_cv_tar_rgb_data, cv_uv_end, 2, cv::Scalar(255,0,0), 2); // blue
            // cv::putText(vis_cv_tar_rgb_data, "E",cv_uv_end, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,0,0), 1.8);
            // cv::imwrite(config_->save_figure_path_ + "/vis_tar_data.png", vis_cv_tar_rgb_data);
            // cv::imshow("vis_tar_data", vis_cv_tar_rgb_data);
            // cv::waitKey(0);

        }


        return true;
    }
    else
    {
        // std::cout << " nothing to match " << std::endl;
        return false;
    }

    

    

}


float MultiViewStereo::SAD(const PixelPoint* ptr_ref_pixel_point_matrix, uint32_t ref_u, uint32_t ref_v, const PixelPoint* ptr_tar_pixel_point_matrix, uint32_t tar_u, uint32_t tar_v, uint32_t width, uint32_t height, int half_ws)
{

    float cost = 0;
    for (int y = -half_ws; y <= half_ws; y++)
    {
        uint32_t ref_row = (ref_v + y)*width;
        uint32_t tar_row = (tar_v + y)*width;
        for (int x = -half_ws; x <= half_ws; x++)
        {
            cost += std::abs(ptr_ref_pixel_point_matrix[ref_row + ref_u + x].intensity_-ptr_tar_pixel_point_matrix[tar_row+tar_u+x].intensity_);
        }
    }


    return cost;

}

float MultiViewStereo::ZSAD(const cv::Mat p1, const cv::Mat p2)
{
    CV_Assert(p1.size() == p2.size());

    // Compute mean intensity of both patches
    cv::Scalar mean1 = cv::mean(p1);
    cv::Scalar mean2 = cv::mean(p2);
    float m2_minus_m1 = (float)mean2[0] - (float)mean1[0];

    float cost = 0.0f;

    for (int y = 0; y < p1.rows; y++) {
        const float* row1 = p1.ptr<float>(y);
        const float* row2 = p2.ptr<float>(y);

        for (int x = 0; x < p1.cols; x++) {
            cost += std::abs(row1[x] - row2[x] + m2_minus_m1);
        }
    }
    return cost;
}


float MultiViewStereo::NCC(const cv::Mat& p1, const cv::Mat& p2) 
{
    CV_Assert(p1.size() == p2.size());

    cv::Scalar mean1, std1, mean2, std2;
    cv::meanStdDev(p1, mean1, std1);
    cv::meanStdDev(p2, mean2, std2);

    float num = 0;
    for (int y = 0; y < p1.rows; y++)
        for (int x = 0; x < p1.cols; x++) {
            float v1 = p1.at<float>(y,x) - (float)mean1[0];
            float v2 = p2.at<float>(y,x) - (float)mean2[0];
            num += v1 * v2;
        }
    return num / ((std1[0]*std2[0] + 1e-6f) * p1.total());
}


float MultiViewStereo::Census(const cv::Mat& p1, const cv::Mat& p2) 
{
    CV_Assert(p1.size() == p2.size());

    int rows = p1.rows;
    int cols = p1.cols;

    // Encode patch1
    uint64_t desc1 = 0;
    float center1 = p1.at<float>(rows/2, cols/2);

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            if (x == cols/2 && y == rows/2) continue; // skip center
            desc1 <<= 1;
            desc1 |= (p1.at<float>(y, x) < center1) ? 1 : 0;
        }
    }

    // Encode patch2
    uint64_t desc2 = 0;
    float center2 = p2.at<float>(rows/2, cols/2);

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            if (x == cols/2 && y == rows/2) continue;
            desc2 <<= 1;
            desc2 |= (p2.at<float>(y, x) < center2) ? 1 : 0;
        }
    }

    // Hamming distance
    uint64_t v = desc1 ^ desc2;
    int cost = 0;
    while (v) {
        cost += v & 1;
        v >>= 1;
    }

    return static_cast<float>(cost);
}
    
}