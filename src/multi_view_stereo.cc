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
        match(ref_image_, tar_image);
        cost_aggregation(ref_image_);
        update_uncertainty(ref_image_);


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

void MultiViewStereo::match(Image* const ref_image, Image* const tar_image)
{

    PixelPoint* ptr_pixel_point_matrix = ref_image->getPixelPointMatrixPtr();

    Eigen::Matrix3f K_ref = config_->cameras_[ref_image->getCameraId()].getIntrinsicsMatrix();
    Eigen::Matrix3f K_tar = config_->cameras_[tar_image->getCameraId()].getIntrinsicsMatrix();
    Eigen::Matrix4f T_world_ref = ref_image->getTransformationMatrix();
    Eigen::Matrix4f T_world_tar = tar_image->getTransformationMatrix();
    Eigen::Matrix4f T_tar_ref =  invertTransform(T_world_tar) * T_world_ref;


    Eigen::Matrix3f R_tar_ref = T_tar_ref.block<3,3>(0,0);
    Eigen::Vector3f t_tar_ref = T_tar_ref.block<3,1>(0,3);
    
    Eigen::Matrix3f KRKi = K_tar * R_tar_ref * K_ref.inverse();
    Eigen::Vector3f Kt = K_tar * t_tar_ref;

    ref_image->setTarImagePtr(tar_image);
    ref_image->setKRKi(KRKi);
    ref_image->setKt(Kt);

    const cv::Mat ref_rgb_data = ref_image->getBGRData();



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

            // if(((tar_image->getId() == 83 || tar_image->getId() == 84)&& u == 75 && v == 238))
            // {
            //     std::cout << "debug" << std::endl;
            // }

            if((u == 1369 && v == 5))
            {
                std::cout << "debug" << std::endl;
            }

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
            bool is_valid_matched_pixel = false;

            if(config_->debug_plot_== true)
            {
                pixel_point.debug_info_vec_.clear();
                pixel_point.debug_info_vec_.resize(pixel_point.epipolar_segment_vec_.size());
            }

            for(int epipolar_segment_idx=0; epipolar_segment_idx<(int)pixel_point.epipolar_segment_vec_.size();epipolar_segment_idx++)
            {
                
                EpipolarSegment& epipolar_segment = pixel_point.epipolar_segment_vec_[epipolar_segment_idx];
                DebugInfo* debug_info_ptr = nullptr;
                if(config_->debug_plot_== true)
                {
                    debug_info_ptr = &pixel_point.debug_info_vec_[epipolar_segment_idx];
                }

                Eigen::Vector3f pt_min = KRKi_uv_homo + epipolar_segment.min_inv_depth_ * Kt;
                Eigen::Vector3f pt_max = KRKi_uv_homo + epipolar_segment.max_inv_depth_ * Kt;

                if(pt_min[2] <=0 || pt_max[2] <= 0){
                    
                    std::cout << "u,v:" << u << "," << v << " 3d point in the back of image" << std::endl;
                    pixel_point.status_ = PixelPoint::Status::INVALID;
                    continue;

                }
                
                Eigen::Vector2f uv_min = Eigen::Vector2f(pt_min[0]/pt_min[2], pt_min[1]/pt_min[2]);
                Eigen::Vector2f uv_max = Eigen::Vector2f(pt_max[0]/pt_max[2], pt_max[1]/pt_max[2]);

                if(config_->debug_plot_ == true)
                {
                    debug_info_ptr->clear();

                    debug_info_ptr->init_min_depth_ = 1/epipolar_segment.max_inv_depth_;
                    debug_info_ptr->init_max_depth_ = 1/epipolar_segment.min_inv_depth_;
                    debug_info_ptr->init_depth_ = (debug_info_ptr->init_min_depth_ + debug_info_ptr->init_max_depth_) / 2.0;
                    debug_info_ptr->uv_min_ = uv_min;
                    debug_info_ptr->uv_max_ = uv_max;

                    
                }

                epipolar_segment.clear();
                Eigen::Vector2f uv_best_match;
                bool is_valid_epipolar_search = epipolarSearch(ptr_pixel_point_matrix, u, v, epipolar_segment_idx, tar_image, uv_min, uv_max, uv_best_match);
                // auto end = std::chrono::steady_clock::now();
                // auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                // printf("epipolar search done ! Timing : %lld µs\n", (long long)elapsed_us);

                // dx*dx > dy*dy
                if(is_valid_epipolar_search == true){
                    is_valid_matched_pixel = true;
                    
    
                }


            }

            if(is_valid_matched_pixel)
            {
                pixel_point.status_ = PixelPoint::Status::UNCERTAINTY_DEPTH;
                for(int epipolar_segment_idx=0; epipolar_segment_idx<(int)pixel_point.epipolar_segment_vec_.size();epipolar_segment_idx++)
                {
                    
                    EpipolarSegment& epipolar_segment = pixel_point.epipolar_segment_vec_[epipolar_segment_idx];
       
                    epipolar_segment.tmp_aggregate_costs_.resize(epipolar_segment.costs_.size(), 0);
                    epipolar_segment.aggregate_costs_.resize(epipolar_segment.costs_.size(), 0);
                    std::copy(epipolar_segment.costs_.begin(), epipolar_segment.costs_.end(), epipolar_segment.aggregate_costs_.begin());
                    
                    for(int i = 0; i < epipolar_segment.valid_uvs_.size();i++)
                    {
                        
                        Eigen::Vector2f valid_uv = epipolar_segment.valid_uvs_[i];

                        float a0 = Kt[0] - Kt[2]*valid_uv[0];
                        float a1 = Kt[1] - Kt[2]*valid_uv[1];
                        float b0 = KRKi_uv_homo[2]*valid_uv[0] - KRKi_uv_homo[0];
                        float b1 = KRKi_uv_homo[2]*valid_uv[1] - KRKi_uv_homo[1];
                        float inv_depth = (a0*b0 + a1*b1) / (a0*a0 + a1*a1);

                        epipolar_segment.inv_depths_.push_back(inv_depth);


                    }


                }
       
            }
            else
            {
                pixel_point.status_ = PixelPoint::Status::INVALID;


                //  if(config_->debug_plot_ == true){
                
                //     debug_info_ptr->clear();

                // }


            }
            
            

            // std::cout << "u,v,d: " << u << "," <<  v <<  "," << ref_ptr_depth_data[v*width+u]  << std::endl;
        }

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

 

bool MultiViewStereo::epipolarSearch(PixelPoint* const ref_ptr_pixel_point_matrix, uint32_t ref_u, uint32_t ref_v, uint32_t epipolar_segment_idx, const Image* const tar_image, 
     Eigen::Vector2f &tar_uv_min, Eigen::Vector2f &tar_uv_max, Eigen::Vector2f& tar_uv_best_match)
{


    uint32_t width = tar_image->getWidth();
    uint32_t height = tar_image->getHeight();  

    Eigen::Vector2f epipolar_vector = Eigen::Vector2f(tar_uv_max[0] - tar_uv_min[0], tar_uv_max[1] - tar_uv_min[1]);
    float epipolar_length = epipolar_vector.norm();

	// ============== check the search distance. everything less than 3px is no need to refine (-> skip). ===================

    // if(epipolar_length <= 2){
    //     return false;
    // }
    Eigen::Vector2f unit_epipolar_vector =  epipolar_vector / epipolar_length;

    if(std::abs(tar_uv_max[0] - tar_uv_min[0]) < 1e-3 || epipolar_length < 1e-3)
    {
        return false;
    }



    bool tar_uv_min_isInImage = tar_image->isInImage(tar_uv_min[0], tar_uv_min[1], half_ws_);
    bool tar_uv_max_isInImage = tar_image->isInImage(tar_uv_max[0], tar_uv_max[1], half_ws_);
    if(tar_uv_min_isInImage == false || tar_uv_max_isInImage == false){
        return false;
    }

    int coord = ref_v * width + ref_u;
    PixelPoint& pixel_point_in_ref_image = ref_ptr_pixel_point_matrix[coord];

    // float k = (tar_uv_max[1] - tar_uv_min[1])/(tar_uv_max[0] - tar_uv_min[0]);
    // float h1 = half_ws_;
    // float h2 = height - half_ws_ - 1;
    // float w1 = half_ws_;
    // float w2 = width - half_ws_ - 1;
    

    // std::vector<Eigen::Vector2f> intersections_in_tar_image;

    // Eigen::Vector2f intersect_y_h1 = Eigen::Vector2f((h1 + k*tar_uv_max[0] - tar_uv_max[1])/k, h1);
    // if(intersect_y_h1[0] >= half_ws_ && intersect_y_h1[0] <= width - half_ws_ - 1)
    // {
    //     intersections_in_tar_image.emplace_back(intersect_y_h1);
    // }
    // Eigen::Vector2f intersect_y_h2 = Eigen::Vector2f((h2 + k*tar_uv_max[0] - tar_uv_max[1])/k, h2);
    // if(intersect_y_h2[0] >= half_ws_ && intersect_y_h2[0] <= width - half_ws_ - 1)
    // {
    //     intersections_in_tar_image.emplace_back(intersect_y_h2);
    // }
    // Eigen::Vector2f intersect_x_w1 = Eigen::Vector2f(w1, k*w1+tar_uv_max[1] - k*tar_uv_max[0]);
    // if(intersect_x_w1[1] >= half_ws_ && intersect_x_w1[1] <= height - half_ws_ - 1)
    // {
    //     intersections_in_tar_image.emplace_back(intersect_x_w1);
    // }
    // Eigen::Vector2f intersect_x_w2 = Eigen::Vector2f(w2, k*w2 + tar_uv_max[1] - k*tar_uv_max[0]);
    // if(intersect_x_w2[1] >= half_ws_ && intersect_x_w2[1] <= height - half_ws_ - 1)
    // {
    //     intersections_in_tar_image.emplace_back(intersect_x_w2);
    // }
  

    // if(intersections_in_tar_image.size() == 0)
    // {
    //     return false;
    // }

    // // case one both in image
    // if(tar_uv_min_isInImage == true && tar_uv_max_isInImage == true)
    // {
    //     //do nothing
    // }
    // else if(tar_uv_min_isInImage == true && tar_uv_max_isInImage == false){


    //     getValidTarUV(tar_uv_max, tar_uv_max, intersections_in_tar_image);


    // }
    // else if(tar_uv_min_isInImage == false && tar_uv_max_isInImage == true)
    // {

    //     getValidTarUV(tar_uv_min, tar_uv_min, intersections_in_tar_image);

    // }
    // else if(tar_uv_min_isInImage == false && tar_uv_max_isInImage == false)
    // {

    //     // tar_uv_min, tar_uv_max were projected in same side.
    //     if(tar_uv_min[0] < w1 && tar_uv_max[0] < w1 || tar_uv_min[0] > w2 && tar_uv_max[0] > w2 || tar_uv_min[1] < h1 && tar_uv_max[1] < h1 || tar_uv_min[1] > h2 && tar_uv_max[1] > h2 )
    //     {
    //         return false;
    //     }

    //     getValidTarUV(tar_uv_min, tar_uv_min, intersections_in_tar_image);
    //     getValidTarUV(tar_uv_max, tar_uv_max, intersections_in_tar_image);

    // }



    // epipolar_vector = Eigen::Vector2f(tar_uv_max[0] - tar_uv_min[0], tar_uv_max[1] - tar_uv_min[1]);
    // epipolar_length = epipolar_vector.norm();
    // unit_epipolar_vector =  epipolar_vector / epipolar_length;

    // pixel_point_in_ref_image.unit_epipolar_vector_ = unit_epipolar_vector;

    PixelPoint* tar_ptr_pixel_point_matrix = tar_image->getPixelPointMatrixPtr();

  

    float s = 0;
    float min_cost = std::numeric_limits<float>::infinity();
    int min_cost_idx = -1;
    float max_cost = -1;
    int max_cost_idx = -1;
    int match_idx = 0;
    int best_s_idx = 0;
    int best_s_idx_temp = 0;

    Eigen::Vector2f uv_best_tmp(-1, -1);
    // float best_s_tmp = 0;
    

    // for debug_plot
    float ws_2 = (2*half_ws_+1)*(2*half_ws_+1);
    EpipolarSegment& epipolar_segment = pixel_point_in_ref_image.epipolar_segment_vec_[epipolar_segment_idx];
    DebugInfo* debug_info_ptr = &pixel_point_in_ref_image.debug_info_vec_[epipolar_segment_idx];

    while(s <= epipolar_length)
    {   
        Eigen::Vector2f tar_uv_current  = tar_uv_min + s * unit_epipolar_vector;
        bool isInImage = tar_image->isInImage(tar_uv_current[0], tar_uv_current[1], half_ws_);

        if(isInImage){
            float cost = SAD(ref_ptr_pixel_point_matrix, ref_u, ref_v, tar_ptr_pixel_point_matrix, round(tar_uv_current[0]), round(tar_uv_current[1]), width, height, half_ws_)/(ws_2);
            epipolar_segment.valid_uvs_.push_back(tar_uv_current);
            epipolar_segment.costs_.push_back(cost);

            if(config_->debug_plot_ == true)
            {   
                debug_info_ptr->steps_.push_back(s);
            }
            
            if(cost < min_cost)
            {
                min_cost = cost;
                uv_best_tmp = tar_uv_current;
                min_cost_idx = match_idx;
                // best_s_tmp = s;

                if(config_->debug_plot_ == true){
                    best_s_idx_temp = match_idx;
                }
            }
            if(cost > max_cost){
                max_cost = cost;
                max_cost_idx = match_idx;
            }

            match_idx+=1;
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
    if(min_cost <= ACCEPTABLE_MINI_COST && 
       min_cost_idx != -1 &&
       max_cost_idx != -1 &&
       min_cost_idx != 0 && 
       min_cost_idx != match_idx - 1 &&
    max_cost - min_cost >= ACCEPTABLE_COST_DIFF){
        tar_uv_best_match = uv_best_tmp;
        pixel_point_in_ref_image.unit_epipolar_vector_ = unit_epipolar_vector;


        return true;
    }
    else
    {
        // std::cout << " nothing to match " << std::endl;
        return false;
    }

    

}


void MultiViewStereo::cost_aggregation(Image* const ref_image)
{
    
    cost_aggregation_left_right(ref_image, true);
    cost_aggregation_left_right(ref_image, false);
    cost_aggregation_up_down(ref_image, true);
    cost_aggregation_up_down(ref_image, false);
    // normalize_aggregated_cost(ref_image);


}

void MultiViewStereo::normalize_aggregated_cost(Image* const ref_image)
{

    size_t height = ref_image->getHeight();
    size_t width = ref_image->getWidth();
    PixelPoint* ptr_pixel_point_matrix = ref_image->getPixelPointMatrixPtr();

    
    for(size_t v=half_ws_; v < height - half_ws_; v++)
    {
        for(size_t u=half_ws_; u < width - half_ws_; u++)
        {

            if(v < config_->start_match_uv_[1] || u < config_->start_match_uv_[0]){
                continue;
            }

            if(u == 1369 && v == 5){
                    std::cout <<"debug" << std::endl;
            }
  

            int current_coord = v*width + u;

            PixelPoint& pixel_point = ptr_pixel_point_matrix[current_coord];

            if(pixel_point.status_ == PixelPoint::Status::INVALID){
                continue;
            }

            float max_aggreagted_cost = 0;
            float min_aggregated_cost = std::numeric_limits<float>::infinity();
            // find out all the local minimum peak
            for(int epipolar_segment_idx = 0 ; epipolar_segment_idx < pixel_point.epipolar_segment_vec_.size(); epipolar_segment_idx++)
            {
                EpipolarSegment& epipolar_segment = pixel_point.epipolar_segment_vec_[epipolar_segment_idx];
                for(int cost_idx = 0; cost_idx < epipolar_segment.aggregate_costs_.size(); cost_idx++)
                {
                    if(max_aggreagted_cost < epipolar_segment.aggregate_costs_[cost_idx]){
                        max_aggreagted_cost = epipolar_segment.aggregate_costs_[cost_idx];
                    }
                    if(min_aggregated_cost > epipolar_segment.aggregate_costs_[cost_idx])
                    {
                        min_aggregated_cost = epipolar_segment.aggregate_costs_[cost_idx];
                    }
                }

            }

            float max_min_aggregated_cost = std::max((max_aggreagted_cost - min_aggregated_cost), (float)1e-3);
            for(int epipolar_segment_idx = 0 ; epipolar_segment_idx < pixel_point.epipolar_segment_vec_.size(); epipolar_segment_idx++)
            {
                EpipolarSegment& epipolar_segment = pixel_point.epipolar_segment_vec_[epipolar_segment_idx];
                for(int cost_idx = 0; cost_idx < epipolar_segment.aggregate_costs_.size(); cost_idx++)
                {
                    epipolar_segment.aggregate_costs_[cost_idx] = (epipolar_segment.aggregate_costs_[cost_idx] - min_aggregated_cost) / (max_min_aggregated_cost);
                }
            }
        }
    }


}

// void MultiViewStereo::getMininumCost(std::vector<std::vector<float>>& aggregated_epipolar_segments_in_a_pos)
// {

//     float mini_cost = std::numeric_limits<float>::infinity();
//     for(int aggregated_epipolar_segment_idx=0; aggregated_epipolar_segment_idx < (int)aggregated_epipolar_segments_in_a_pos.size(); aggregated_epipolar_segment_idx++)
//     {

//         std::vector<float>& aggregated_costs = aggregated_epipolar_segments_in_a_pos[aggregated_epipolar_segment_idx];
//         for(int cost_idx = 0; cost_idx < (int)aggregated_costs.size(); cost_idx++ )
//         {
//             if(aggregated_costs[cost_idx] < mini_cost)
//             {
//                 mini_cost = aggregated_costs[cost_idx];
//             } 
//         }
//     }

//     return mini_cost;

// }

// void MultiViewStereo::getInterpolatedAggregatedCostByInvDepth(std::vector<std::vector<float>>& aggregated_epipolar_segments_in_a_pos)
// {
//     bool is_exist_interpolated_cost = false;
//     float interpolated_cost = std::numeric_limits<float>::infinity();
//     for(int aggregated_epipolar_segment_idx = 0; aggregated_epipolar_segment_idx < (int)aggregated_epipolar_segments_in_a_pos.size(); aggregated_epipolar_segment_idx++)
//     {
        
//         std::vector<float>& aggregated_costs = aggregated_epipolar_segments_in_a_pos[aggregated_epipolar_segment_idx];

//         for(int inv_depth_idx = 1; inv_depth_idx < (int)epipolar_segment.inv_depths_.size(); inv_depth_idx++)
//         {
//             float lower_inv_depth = epipolar_segment.inv_depths_[inv_depth_idx-1];
//             float inv_depth = epipolar_segment.inv_depths_[inv_depth_idx];
//             if(ref_inv_depth <= inv_depth && ref_inv_depth >= lower_inv_depth)
//             {
//                 interpolated_cost =  epipolar_segment.costs_[inv_depth_idx-1] + (inv_depth - lower_inv_depth) * (epipolar_segment.costs_[inv_depth_idx] - epipolar_segment.costs_[inv_depth_idx-1])/(inv_depth - lower_inv_depth);
//                 is_exist_interpolated_cost = true;
//                 break;
//             }
//         }

//         if(is_exist_interpolated_cost == true)
//         {
//             break;
//         }
//     }

//     return interpolated_cost;

// }



void MultiViewStereo::cost_aggregation_left_right(Image* const ref_image, bool is_forward)
{

    size_t height = ref_image->getHeight();
    size_t width = ref_image->getWidth();
    PixelPoint* ptr_pixel_point_matrix = ref_image->getPixelPointMatrixPtr();
	const int direction = is_forward ? 1 : -1;

    for(size_t y = half_ws_; y < height - half_ws_; y++)
    {
        bool is_init_aggregate_path = false;
        size_t v = y;

        // forward aggregate
        for(size_t x = half_ws_; x < width - half_ws_; x++)
        {
            size_t u = (is_forward) ? x : width - x - 1;

            if(v < config_->start_match_uv_[1] || u < config_->start_match_uv_[0]){
                continue;
            }

            if((u == 1369 && v == 5))
            {
                std::cout << "debug" << std::endl;
            }

            int current_coord = v*width + u;
            PixelPoint& pixel_point = ptr_pixel_point_matrix[current_coord];

            if(pixel_point.status_ == PixelPoint::Status::INVALID){
                is_init_aggregate_path = false;
                continue;
            }


            // init aggregate path
            if(is_init_aggregate_path == false)
            {
                init_aggregate_pixel(pixel_point);
                is_init_aggregate_path = true;
                continue;
            }

            // forward update
            PixelPoint& pixel_point_in_last_pos = ptr_pixel_point_matrix[current_coord-direction];

            if(cost_aggregation_from_last_pixel(pixel_point_in_last_pos, pixel_point) == true)
            {

            }
            // fail in aggregation, then just copy cost to tmp aggregate cost
            else{
                init_aggregate_pixel(pixel_point);

            }


        }

    }

}

void MultiViewStereo::cost_aggregation_up_down(Image* const ref_image, bool is_forward)
{

    size_t height = ref_image->getHeight();
    size_t width = ref_image->getWidth();
    PixelPoint* ptr_pixel_point_matrix = ref_image->getPixelPointMatrixPtr();
	const int direction = is_forward ? 1 : -1;

    for(size_t x = half_ws_; x < width - half_ws_; x++)
    {
        bool is_init_aggregate_path = false;
        size_t u = x;

        // forward aggregate
        for(size_t y = half_ws_; y < height - half_ws_; y++)
        {
            size_t v = (is_forward) ? y : height - y - 1;

            if(v < config_->start_match_uv_[1] || u < config_->start_match_uv_[0]){
                continue;
            }

            // if((is_forward == false) && (v == 20))
            // {
            //     std::cout << "debug" << std::endl;
            // }

            int current_coord = v*width + u;
            PixelPoint& pixel_point = ptr_pixel_point_matrix[current_coord];

            if(pixel_point.status_ == PixelPoint::Status::INVALID){
                is_init_aggregate_path = false;
                continue;
            }


            // init aggregate path
            if(is_init_aggregate_path == false)
            {
                init_aggregate_pixel(pixel_point);
                is_init_aggregate_path = true;
                continue;
            }

            // forward update
            PixelPoint& pixel_point_in_last_pos = ptr_pixel_point_matrix[current_coord-direction*width];
            
            if(cost_aggregation_from_last_pixel(pixel_point_in_last_pos, pixel_point) == true)
            {

            }
            // fail in aggregation, then just copy cost to tmp aggregate cost
            else{
                init_aggregate_pixel(pixel_point);

            }

        }

    }

}



bool MultiViewStereo::cost_aggregation_from_last_pixel(const PixelPoint& pixel_point_in_last_pos, PixelPoint& pixel_point)
{

    int minimum_cost_epipolar_segment_idx = -1;
    int minimum_cost_idx = -1;
    pixel_point_in_last_pos.getMininumIdxTmpAggregatedCost(minimum_cost_epipolar_segment_idx, minimum_cost_idx);

    if(pixel_point.DoesExistDepthIntersection(pixel_point_in_last_pos, minimum_cost_epipolar_segment_idx) == true)
    {

        float mini_tmp_aggregated_cost_in_last_pos = pixel_point_in_last_pos.epipolar_segment_vec_[minimum_cost_epipolar_segment_idx].tmp_aggregate_costs_[minimum_cost_idx];
        float inv_depth_in_last_pos_in_at_mini_cost = pixel_point_in_last_pos.epipolar_segment_vec_[minimum_cost_epipolar_segment_idx].inv_depths_[minimum_cost_idx];
        
        float depth_in_last_pos_in_at_mini_cost = 1.0f/inv_depth_in_last_pos_in_at_mini_cost;
        float acceptable_depth = depth_in_last_pos_in_at_mini_cost * ACCEPTABLE_DEPTH_PARAMETER;
        float depth_minus_in_last_pos_in_at_mini_cost = depth_in_last_pos_in_at_mini_cost - acceptable_depth;
        float inv_depth_minus_in_last_pos_in_at_mini_cost = 1/depth_minus_in_last_pos_in_at_mini_cost;
        float depth_plus_in_last_pos_in_at_mini_cost = depth_in_last_pos_in_at_mini_cost + acceptable_depth;
        float inv_depth_plus_in_last_pos_in_at_mini_cost = 1/depth_plus_in_last_pos_in_at_mini_cost;

        for(int epipolar_segment_idx=0; epipolar_segment_idx<(int)pixel_point.epipolar_segment_vec_.size(); epipolar_segment_idx++)
        {

            EpipolarSegment& epipolar_segment = pixel_point.epipolar_segment_vec_[epipolar_segment_idx];

            for(int cost_idx = 0; cost_idx < (int)epipolar_segment.costs_.size(); cost_idx++)
            {

                float inv_depth = epipolar_segment.inv_depths_[cost_idx];
                float depth = 1/inv_depth;
                float tmp_aggregated_cost_in_last_pos = mini_tmp_aggregated_cost_in_last_pos;

                // if(cost_idx == 1)
                // {
                //     std::cout << "debug" << std::endl;
                // }

                // if(depth <= depth_minus_in_last_pos_in_at_mini_cost){


                //     tmp_aggreagted_cost_in_last_pos = pixel_point_in_last_pos.getInterpolatedTmpAggregatedCostByInvDepth(1.0f/(depth+acceptable_depth));

                // }
                // else if(depth >= depth_plus_in_last_pos_in_at_mini_cost)
                // {

                //     tmp_aggreagted_cost_in_last_pos = pixel_point_in_last_pos.getInterpolatedTmpAggregatedCostByInvDepth(1.0f/(depth-acceptable_depth));

                // }

                tmp_aggregated_cost_in_last_pos = pixel_point_in_last_pos.getInterpolatedTmpAggregatedCostByInvDepth(inv_depth);


                // if(tmp_aggreagted_cost_in_last_pos < 0)
                // {
                //     std::cout << "debug" << std::endl;
                // }

                float tmp_aggregate_cost = aggregate_cost_func(
                    epipolar_segment.costs_[cost_idx], inv_depth, pixel_point.intensity_, 
                    tmp_aggregated_cost_in_last_pos, mini_tmp_aggregated_cost_in_last_pos, inv_depth_minus_in_last_pos_in_at_mini_cost,
                    pixel_point_in_last_pos.intensity_);

                // if(tmp_aggregate_cost < 0){
                //     std::cout << "debug" << std::endl;
                //     float tmp_tmp_aggregate_cost = aggregate_cost_func(
                //     epipolar_segment.costs_[cost_idx], inv_depth, pixel_point.intensity_, 
                //     tmp_aggregated_cost_in_last_pos, mini_tmp_aggregated_cost_in_last_pos, inv_depth_minus_in_last_pos_in_at_mini_cost,
                //     pixel_point_in_last_pos.intensity_);
                // }
                    
                epipolar_segment.tmp_aggregate_costs_[cost_idx] = tmp_aggregate_cost;
                epipolar_segment.aggregate_costs_[cost_idx] += epipolar_segment.tmp_aggregate_costs_[cost_idx]; 
            }
            
        }

        return true;

    }
    else
    {

        return false;

    }
    
    

}

float MultiViewStereo::aggregate_cost_func(float cost_in_curr_pos, float inv_depth_in_curr_pos, float intensity_in_curr_pos, 
        float tmp_aggreagted_cost_in_last_pos, float mini_tmp_aggregated_cost_in_last_pos, float inv_depth_in_last_pos_in_at_mini_cost, float intensity_in_last_pos)
{

    float depth_in_curr_pos = 1/inv_depth_in_curr_pos;

    float most_likely_depth_in_last_pos = 1/inv_depth_in_last_pos_in_at_mini_cost;

    // float P = std::max(threshold, alpha * mini_inv_depth_in_last_pos / (std::abs(intensity_in_curr_pos - intensity_in_last_pos) + 1));
    float P = MAXIMUM_AGGREAGTE_COST_PENALTY*mini_tmp_aggregated_cost_in_last_pos/(std::abs(intensity_in_curr_pos - intensity_in_last_pos) + 1);
    // float alpha = 10.0f;

    // float depth_diff = std::abs(depth_in_curr_pos - depth_in_last_pos);
    // float P = alpha * std::exp(-depth_in_last_pos) * (depth_diff );
    // float aggregate_cost = cost_in_curr_pos + std::min(tmp_aggreagted_cost_in_last_pos, mini_tmp_aggregated_cost_in_last_pos + P) - mini_tmp_aggregated_cost_in_last_pos;

    // float aggregate_cost = cost_in_curr_pos + std::min(tmp_aggreagted_cost_in_last_pos, mini_tmp_aggregated_cost_in_last_pos + P) - mini_tmp_aggregated_cost_in_last_pos;

    // float aggregate_cost = cost_in_curr_pos + std::min( 
    //     std::min(tmp_aggreagted_cost_in_last_pos, tmp_aggreagted_cost_minus_in_last_pos + P_minus),
    //     std::min(tmp_aggreagted_cost_plus_in_last_pos + P_plus, mini_tmp_aggregated_cost_in_last_pos + P)
    //     ) - mini_tmp_aggregated_cost_in_last_pos;

    float aggregate_cost = cost_in_curr_pos + std::min( 
        tmp_aggreagted_cost_in_last_pos - mini_tmp_aggregated_cost_in_last_pos, P
        );

    // float aggregate_cost = cost_in_curr_pos;
    // if(std::isinf(tmp_aggreagted_cost_in_last_pos))
    // {
    //     aggregate_cost = cost_in_curr_pos;
    // }
    // else
    // {
    //     aggregate_cost = cost_in_curr_pos + tmp_aggreagted_cost_in_last_pos - mini_tmp_aggregated_cost_in_last_pos;
  
    // }
    

    return aggregate_cost;
}



void MultiViewStereo::init_aggregate_pixel(PixelPoint& pixel_point)
{

    // init aggregate pixel in aggregate path
    // aggregate_epipolar_segments_in_curr_pos.resize(pixel_point.epipolar_segment_vec_.size());
    for(int epipolar_segment_idx=0; epipolar_segment_idx<(int)pixel_point.epipolar_segment_vec_.size();epipolar_segment_idx++)
    {

        EpipolarSegment& epipolar_segment = pixel_point.epipolar_segment_vec_[epipolar_segment_idx];
        // std::vector<float>& aggregate_costs = aggregate_epipolar_segments_in_curr_pos[epipolar_segment_idx];

        for(int cost_idx = 0; cost_idx < epipolar_segment.costs_.size(); cost_idx++){
            epipolar_segment.tmp_aggregate_costs_[cost_idx] = epipolar_segment.costs_[cost_idx];
            // epipolar_segment.aggregate_costs_[cost_idx] += epipolar_segment.costs_[cost_idx];
        }

    }


}




void MultiViewStereo::update_uncertainty(Image* const ref_image)
{

    float const_error_in_pixel = 1.0f;

    Eigen::Matrix3f KRKi = ref_image->getKRKi();
    Eigen::Vector3f Kt = ref_image->getKt();

    uint32_t height = ref_image->getHeight();
    uint32_t width = ref_image->getWidth();

    PixelPoint* ptr_pixel_point_matrix = ref_image->getPixelPointMatrixPtr();


    for(size_t v=half_ws_; v < height - half_ws_; v++)
    {
        for(size_t u=half_ws_; u < width - half_ws_; u++)
        {

            if(v < config_->start_match_uv_[1] || u < config_->start_match_uv_[0]){
                continue;
            }

            if(u == 113 && v == 191){
                std::cout <<"debug" << std::endl;
            }
            // std::cout << "debug " << " u= " << u << ", v= " << v << std::endl;

            int current_coord = v*width + u;

            PixelPoint& pixel_point = ptr_pixel_point_matrix[current_coord];

            if(pixel_point.status_ == PixelPoint::Status::INVALID){
                continue;
            }
                
            DebugInfo* debug_info_ptr = nullptr;
            if(config_->debug_plot_ == true){
                debug_info_ptr = &pixel_point.debug_info_vec_[0];
            }

            if(config_->debug_plot_ == true)
            {
                debug_info_ptr->epipolar_segment_ = pixel_point.epipolar_segment_vec_[0];
            }

            Eigen::Vector3f KRKi_uv_homo = KRKi * Eigen::Vector3f(u,v,1);

            // find out all the local minimum peak
            for(int epipolar_segment_idx = 0 ; epipolar_segment_idx < pixel_point.epipolar_segment_vec_.size(); epipolar_segment_idx++)
            {
                EpipolarSegment& epipolar_segment = pixel_point.epipolar_segment_vec_[epipolar_segment_idx];
                if(epipolar_segment.aggregate_costs_.size() == 0){
                    continue;
                }
                get_peaks_with_persistent_homology(epipolar_segment.aggregate_costs_, epipolar_segment.local_minimum_peaks_);

            }

            int global_minimum_peak_epipolar_segment_idx = -1;
            int global_minimum_peak_idx = -1;
            pixel_point.getGlobalMininumPeakIdx(global_minimum_peak_epipolar_segment_idx, global_minimum_peak_idx);
            MinimumPeak& global_minimum_peak = pixel_point.epipolar_segment_vec_[global_minimum_peak_epipolar_segment_idx].local_minimum_peaks_[global_minimum_peak_idx];
            float min_aggregate_cost = pixel_point.epipolar_segment_vec_[global_minimum_peak_epipolar_segment_idx].aggregate_costs_[global_minimum_peak.born_idx_];
            float mini_aggregate_cost_idx = global_minimum_peak.born_idx_;
            int mini_aggregate_cost_idx_plus_1 = global_minimum_peak.born_idx_ + 1;
            int mini_aggregate_cost_idx_minus_1 = global_minimum_peak.born_idx_ - 1;

            float acceptable_threshold = min_aggregate_cost + ACCEPTABLE_MINI_COST/10;

            // when there are at least one aggregated cost point in plus and minus direction, 
            // depth uncertainty will be considered to reduce

            if( mini_aggregate_cost_idx_minus_1 >= 0 && mini_aggregate_cost_idx_plus_1 < pixel_point.epipolar_segment_vec_[global_minimum_peak_epipolar_segment_idx].aggregate_costs_.size())
            {
                float min_aggregate_cost_plus_1 = pixel_point.epipolar_segment_vec_[global_minimum_peak_epipolar_segment_idx].aggregate_costs_[mini_aggregate_cost_idx_plus_1];
                float min_aggregate_cost_minus_1 = pixel_point.epipolar_segment_vec_[global_minimum_peak_epipolar_segment_idx].aggregate_costs_[mini_aggregate_cost_idx_minus_1];
                float alpha2 = min_aggregate_cost_minus_1 + min_aggregate_cost_plus_1 - 2 * min_aggregate_cost;


                float minus_beta2 = min_aggregate_cost_minus_1 - min_aggregate_cost_plus_1;
                float beta_square = minus_beta2 * minus_beta2;
                mini_aggregate_cost_idx = mini_aggregate_cost_idx + (minus_beta2)/ (2.0f * alpha2);
                min_aggregate_cost = min_aggregate_cost - (minus_beta2) * (minus_beta2) / (8.0f* alpha2);

                acceptable_threshold = min_aggregate_cost + ACCEPTABLE_MINI_COST/10;
                float root_equation = sqrt(beta_square + 2 * alpha2 * ACCEPTABLE_MINI_COST/10);

                float mini_aggregate_cost_idx_plus = global_minimum_peak.born_idx_ + 1;

                if(min_aggregate_cost_plus_1 >= acceptable_threshold)
                {
                    mini_aggregate_cost_idx_plus = mini_aggregate_cost_idx + (minus_beta2/2 + root_equation)/alpha2;
                }
                else
                {
                    for(int cost_i = mini_aggregate_cost_idx_plus_1 + 1; cost_i < pixel_point.epipolar_segment_vec_[0].aggregate_costs_.size(); cost_i++)
                    {
                        float curr_cost = pixel_point.epipolar_segment_vec_[0].aggregate_costs_[cost_i];
                        float last_cost = pixel_point.epipolar_segment_vec_[0].aggregate_costs_[cost_i-1];

                        if(curr_cost > acceptable_threshold && last_cost <= acceptable_threshold)
                        {
                            mini_aggregate_cost_idx_plus = (float)(cost_i-1) + (acceptable_threshold - last_cost)/(curr_cost - last_cost);
                            break;
                        }
                    }

                }


                float mini_aggregate_cost_idx_minus = global_minimum_peak.born_idx_ - 1;

                if(min_aggregate_cost_minus_1 >= acceptable_threshold)
                {
                    mini_aggregate_cost_idx_minus = mini_aggregate_cost_idx + (minus_beta2/2 - root_equation)/alpha2;

                }
                else
                {
                    for(int cost_i = mini_aggregate_cost_idx_minus_1 - 1; cost_i >= 0; cost_i--)
                    {

                        float curr_cost = pixel_point.epipolar_segment_vec_[0].aggregate_costs_[cost_i];
                        float last_cost = pixel_point.epipolar_segment_vec_[0].aggregate_costs_[cost_i+1];

                        if(curr_cost > acceptable_threshold && last_cost <= acceptable_threshold)
                        {
                            mini_aggregate_cost_idx_minus = (float)(cost_i+1) - (acceptable_threshold - last_cost)/(curr_cost - last_cost);
                            break;
                        }
                    }
                }


                bool isUniquenessUncertaintyRegion = true;
                for(int cost_i = 0; cost_i < pixel_point.epipolar_segment_vec_[0].aggregate_costs_.size(); cost_i++)
                {
                    if( cost_i >= mini_aggregate_cost_idx_minus && cost_i <= mini_aggregate_cost_idx_plus)
                    {
                        continue;
                    }
                    
                    float curr_cost = pixel_point.epipolar_segment_vec_[0].aggregate_costs_[cost_i];

                    if(curr_cost <=  acceptable_threshold)
                    {
                        //reject this uncertainty reduce
                        isUniquenessUncertaintyRegion = false;
                        break;
                    }

                }
                if(isUniquenessUncertaintyRegion == false)
                {
                    continue;
                }



                Eigen::Vector2f uv_possible_match = pixel_point.epipolar_segment_vec_[0].valid_uvs_[0] + mini_aggregate_cost_idx * pixel_point.unit_epipolar_vector_;
                Eigen::Vector2f uv_possible_match_minus = pixel_point.epipolar_segment_vec_[0].valid_uvs_[0] + mini_aggregate_cost_idx_minus * pixel_point.unit_epipolar_vector_;
                Eigen::Vector2f uv_possible_match_plus = pixel_point.epipolar_segment_vec_[0].valid_uvs_[0] + mini_aggregate_cost_idx_plus * pixel_point.unit_epipolar_vector_;


                // Eigen::Vector2f uv_possible_match_minus = uv_possible_match - const_error_in_pixel*pixel_point.unit_epipolar_vector_;
                // Eigen::Vector2f uv_possible_match_plus = uv_possible_match + const_error_in_pixel*pixel_point.unit_epipolar_vector_;

                float min_inv_depth, max_inv_depth = -1.0f;
                pixel_depth_estimation(min_inv_depth, max_inv_depth, Kt, KRKi_uv_homo, uv_possible_match_minus, uv_possible_match_plus);
                pixel_point.epipolar_segment_vec_[0].setMaxInvDepth(max_inv_depth);
                pixel_point.epipolar_segment_vec_[0].setMinInvDepth(min_inv_depth);
                pixel_point.epipolar_segment_vec_[0].output_data_.depth_ = (pixel_point.epipolar_segment_vec_[0].min_depth_ + pixel_point.epipolar_segment_vec_[0].max_depth_) / 2.0f;

        

                if(config_->debug_plot_ == true){

                    debug_info_ptr->possible_minimum_peak_idxes_.emplace_back(round(global_minimum_peak.born_idx_));
                    debug_info_ptr->errors_in_minus_direction_.emplace_back(floor(mini_aggregate_cost_idx_minus));
                    debug_info_ptr->errors_in_plus_direction_.emplace_back(ceil(mini_aggregate_cost_idx_plus));

                    debug_info_ptr->uv_possible_matches_.emplace_back(uv_possible_match);

                    debug_info_ptr->updated_min_depths_.emplace_back(pixel_point.epipolar_segment_vec_[0].min_depth_);
                    debug_info_ptr->updated_max_depths_.emplace_back(pixel_point.epipolar_segment_vec_[0].max_depth_);
                    debug_info_ptr->updated_depths_.emplace_back(pixel_point.epipolar_segment_vec_[0].output_data_.depth_);

                    // std::cout << "min_depth : " << min_depth << " and max_depth : " << max_depth << " at uv_best_match (" << uv_best_match[0] << "," << uv_best_match[1] << ")" << std::endl;

                }

            }


        }

    }
}


void MultiViewStereo::pixel_depth_estimation(float& min_inv_depth, float& max_inv_depth, const Eigen::Vector3f& Kt, const Eigen::Vector3f& KRKi_uv_homo, const Eigen::Vector2f& uv_best_match_minus, const Eigen::Vector2f& uv_best_match_plus)
{

        float a0_minus = (Kt[0] - Kt[2]*(uv_best_match_minus[0]));
        float a1_minus = (Kt[1] - Kt[2]*(uv_best_match_minus[1]));
        float b0_minus = (KRKi_uv_homo[2]*(uv_best_match_minus[0]) - KRKi_uv_homo[0]);
        float b1_minus = (KRKi_uv_homo[2]*(uv_best_match_minus[1]) - KRKi_uv_homo[1]);

        min_inv_depth = (a0_minus*b0_minus + a1_minus*b1_minus) / (a0_minus*a0_minus + a1_minus*a1_minus);


        float a0_plus = (Kt[0] - Kt[2]*(uv_best_match_plus[0]));
        float a1_plus = (Kt[1] - Kt[2]*(uv_best_match_plus[1]));
        float b0_plus = (KRKi_uv_homo[2]*(uv_best_match_plus[0]) - KRKi_uv_homo[0]);
        float b1_plus = (KRKi_uv_homo[2]*(uv_best_match_plus[1]) - KRKi_uv_homo[1]);
        
        max_inv_depth = (a0_plus*b0_plus + a1_plus*b1_plus) / (a0_plus*a0_plus + a1_plus*a1_plus);
        
        if(max_inv_depth <= 0)
        {
            max_inv_depth = 1.0/config_->min_depth_;
        }
        if(min_inv_depth <= 0)
        {
            min_inv_depth = config_->infinite_inv_depth_;
        }
        
        if(min_inv_depth > max_inv_depth) std::swap<float>(min_inv_depth, max_inv_depth);


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