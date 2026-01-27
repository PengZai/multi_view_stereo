#include "multi_view_stereo.h"

namespace MVS
{


MultiViewStereo::MultiViewStereo(Config* const config)
:ref_image_(nullptr),
status_(Status::FREE)
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

void MultiViewStereo::setTargetImage(Image* const image)
{
    tar_image_ = image;
}


Image* MultiViewStereo::getReferenceImage()
{
    return ref_image_;
}

Image* MultiViewStereo::getTargetImage()
{
    return tar_image_;
}

MultiViewStereo::Status MultiViewStereo::getStatus()
{
    return status_;
}

void MultiViewStereo::run()
{

    bool isSuccess = false;
    Image* ref_image = nullptr;
    getImageWithPoseIdAndCameraId(ref_image, config_->ref_pose_idx_, config_->ref_camera_idx_);

    if(ref_image == nullptr){
        std::cout << "ref_image is not existed for ref_pose_id:" << config_->ref_pose_idx_ << " and ref_camera_id:" << config_->ref_camera_idx_ << std::endl;
        return;
    }

    isSuccess = ref_image->loadData();
    if(isSuccess == false){
        return;
    }

    setReferenceImage(ref_image);

    Image* tar_image = nullptr;
    getImageWithPoseIdAndCameraId(tar_image, config_->tar_pose_start_idx_, config_->tar_camera_start_idx_);
    
    if(tar_image == nullptr){
        std::cout << "tar_image is not existed for tar_pose_id:" << config_->tar_pose_start_idx_ << " and tar_camera_id:" << config_->tar_camera_start_idx_ << std::endl;
        return;
    }

    if(ref_image->getId() == tar_image->getId())
    {
        int tar_image_idx = tar_image->getId();
        std::cout << "tar image id is equal to reference frame id :" << tar_image_idx << " , let's us pick next image for target image" << std::endl;
        tar_image_idx++;
        std::vector<Image*>& images = dataset_->getImages();
        if(tar_image_idx >= images.size())
        {
            std::cout << "tar_image_idx is out of range of the number of image" << std::endl;
            return;
        }
        tar_image = images[tar_image_idx];
    }

    isSuccess = tar_image->loadData();        
    if(isSuccess == false){
        return;
    }
    setTargetImage(tar_image);


    visualizer_->setSelectedRefImageId(ref_image->getId());
    visualizer_->setSelectedTarImageId(tar_image->getId());


        
    while (1) {
        
        if(isRunSingleDepthReconstruction_)
        {
            single_depth_reconstruction();
        }   

        if(isRunSceneReconstruction_)
        {
            scene_reconstruction();

        }

        // std::cout << "outside loop" << std::endl;
        visualizer_->showInterface();
        // std::cout << "test" << std::endl;

    }


}



void MultiViewStereo::setIsRunSingleDepthReconstruction(bool isRunSingleDepthReconstruction)
{
    isRunSingleDepthReconstruction_ = isRunSingleDepthReconstruction;
}

void MultiViewStereo::setIsRunSceneReconstruction(bool isRunSceneReconstruction)
{
    isRunSceneReconstruction_ = isRunSceneReconstruction;

}


void MultiViewStereo::getImageWithPoseIdAndCameraId(Image*& ptr_image, int inp_pose_id, int inp_camera_id)
{
    std::vector<Image*>& images = dataset_->getImages();

    for(size_t image_idx=0; image_idx < images.size(); image_idx++){
        Image* image = images[image_idx];
        int pose_id = image->getPoseId();
        int camera_id = image->getCameraId();

        if(pose_id == inp_pose_id && camera_id == inp_camera_id)
        {
            ptr_image = image;
            break;
        }
    }
}

void MultiViewStereo::scene_reconstruction_init()
{
    status_ =  Status::BUSY;
    setReferenceImage(nullptr);
    setTargetImage(nullptr);
    std::cout << "init for scene reconstruction and set it to busy" << std::endl; 
}

void MultiViewStereo::scene_reconstruction()
{

    std::vector<Image*>& images = dataset_->getImages();
    if(ref_image_ == nullptr)
    {
        setReferenceImage(images[0]);
    }

    Eigen::Matrix4f T_world_ref = ref_image_->getTransformationMatrix();

    Image* tar_image = nullptr;
    float minimum_norm_t_tar_ref = std::numeric_limits<float>::infinity();
    for(int idx=0; idx < images.size(); idx++)
    {
        Image* image = images[idx];
        if(image->getId() == ref_image_->getId())
        {
            continue;
        }

        std::map<int, Image*>& map_of_images_has_been_matched = ref_image_->getMapOfImagesHasBeenMatched();
        auto it = map_of_images_has_been_matched.find(image->getId());
        if(it != map_of_images_has_been_matched.end())
        {
            // this Id has been used
            continue;
        }

        Eigen::Matrix4f T_world_tar = image->getTransformationMatrix();
        Eigen::Matrix4f T_tar_ref =  invertTransform(T_world_tar) * T_world_ref;
        Eigen::Vector3f t_tar_ref = T_tar_ref.block<3,1>(0,3);
        double norm_t_tar_ref = t_tar_ref.norm();
        if(norm_t_tar_ref > MAXIMUM_RECONSTRUCTION_DISTANCE)
        {
            continue;
        }

        if(minimum_norm_t_tar_ref > norm_t_tar_ref)
        {
            minimum_norm_t_tar_ref = norm_t_tar_ref;
            tar_image = image;
        }
        
    }

    if(tar_image == nullptr){
        
        std::cout << "single depth reconstuction done for image id " << ref_image_->getId() << std::endl;

        int new_ref_image_id = ref_image_->getId()+1;
        if(new_ref_image_id >= images.size())
        {
            // consistency_check();
            status_ =  Status::FREE;
            setIsRunSceneReconstruction(false);

            std::cout << "scene reconstruction done and set it to free" << std::endl;
            return;
        }
        Image* ref_image = images[new_ref_image_id];
        setReferenceImage(ref_image);
        std::cout << " now set ref image id is " << ref_image_->getId() << std::endl;
        visualizer_->setIsNextAction(true);
        return;

    }

    std::cout << "minimum_norm_t_tar_ref : " << minimum_norm_t_tar_ref << std::endl;

    setTargetImage(tar_image);
    
    single_depth_reconstruction();
    visualizer_->setIsNextAction(true);

}

void MultiViewStereo::single_depth_reconstruction()
{   

    bool isSuccess = false;
    setIsRunSingleDepthReconstruction(false);

    printf("match start for ref pose %d, cam %d and tar pose %d, cam %d\n", ref_image_->getPoseId(), ref_image_->getCameraId(), tar_image_->getPoseId(), tar_image_->getCameraId());
    
    std::map<int, Image*>& map_of_images_has_been_matched = ref_image_->getMapOfImagesHasBeenMatched();
    auto it = map_of_images_has_been_matched.find(tar_image_->getId());
    if(it != map_of_images_has_been_matched.end())
    {
        std::cout << "tar image Id " << std::to_string(tar_image_->getId()) << " has been matched for ref image" << std::to_string(ref_image_->getId()) << std::endl;
        return;
    }

    isSuccess = ref_image_->loadData();
    if(isSuccess == false){
        return;
    }

    isSuccess = tar_image_->loadData();        
    if(isSuccess == false){
        return;
    }

    auto start = std::chrono::steady_clock::now();        
    match(ref_image_, tar_image_);
    cost_aggregation(ref_image_);
    update_uncertainty(ref_image_);


    auto end = std::chrono::steady_clock::now();
    auto tt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("match Done! Timing : %lf s for ref pose %d, cam %d and tar pose %d, cam %d\n", tt.count() / 1000.0, ref_image_->getPoseId(), ref_image_->getCameraId(), tar_image_->getPoseId(), tar_image_->getCameraId());

    map_of_images_has_been_matched[tar_image_->getId()] = tar_image_;
    std::cout << "tar image " << tar_image_->getId() << " is added to map_of_images_has_been_matched of ref image " << ref_image_->getId() << std::endl;

    // visualizer_->saveDepth(ref_image_);        
    
    // visualizer_->showUndistortedGrayImage(ref_image_, "RefUndistortedGray");
    // visualizer_->showUndistortedGrayImage(tar_image, "TarUndistortedGray");

    // visualizer_->saveDepth(ref_image_);

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

    for(int v=half_ws_; v < height - half_ws_; v++)
    {
        for(int u=half_ws_; u < width - half_ws_; u++)
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
            if(pixel_point.status_ == PixelPoint::Status::DISABLE)
            {
                continue;
            }

            bool is_valid_matched_pixel = false;

                
            DebugInfo* debug_info_ptr = nullptr;
            if(config_->debug_plot_== true)
            {
                debug_info_ptr = &pixel_point.debug_info_;
            }

            Eigen::Vector3f pt_min = KRKi_uv_homo + pixel_point.min_inv_depth_ * Kt;
            Eigen::Vector3f pt_max = KRKi_uv_homo + pixel_point.max_inv_depth_ * Kt;

            if(pt_min[2] <=0 || pt_max[2] <= 0){
                
                // std::cout << "u,v:" << u << "," << v << " 3d point in the back of image" << std::endl;
                pixel_point.status_ = PixelPoint::Status::INVALID_COST_OBSERVATION;
                continue;

            }
            
            Eigen::Vector2f uv_min = Eigen::Vector2f(pt_min[0]/pt_min[2], pt_min[1]/pt_min[2]);
            Eigen::Vector2f uv_max = Eigen::Vector2f(pt_max[0]/pt_max[2], pt_max[1]/pt_max[2]);

            if(config_->debug_plot_ == true)
            {
                debug_info_ptr->clear();

                debug_info_ptr->init_min_depth_ = 1/pixel_point.max_inv_depth_;
                debug_info_ptr->init_max_depth_ = 1/pixel_point.min_inv_depth_;
                debug_info_ptr->init_depth_ = (debug_info_ptr->init_min_depth_ + debug_info_ptr->init_max_depth_) / 2.0;
                debug_info_ptr->uv_min_ = uv_min;
                debug_info_ptr->uv_max_ = uv_max;

                
            }

            // clean previous match information before epipolar seach
            pixel_point.clearMatchInformation();

            bool is_valid_epipolar_search = epipolarSearch(ref_image, u, v, tar_image, uv_min, uv_max);
            // auto end = std::chrono::steady_clock::now();
            // auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            // printf("epipolar search done ! Timing : %lld µs\n", (long long)elapsed_us);

            // dx*dx > dy*dy
            if(is_valid_epipolar_search == true){
                is_valid_matched_pixel = true;
                pixel_point.status_ = PixelPoint::Status::NORMAL;

            }


            

            if(is_valid_matched_pixel)
            {         
                pixel_point.tmp_aggregate_costs_.resize(pixel_point.costs_.size(), 0);
                pixel_point.aggregate_costs_.resize(pixel_point.costs_.size(), 0);
                pixel_point.num_has_been_aggregated_ = 0;

                for(int i = 0; i < pixel_point.valid_uvs_.size();i++)
                {
                    
                    Eigen::Vector2f valid_uv = pixel_point.valid_uvs_[i];

                    float a0 = Kt[0] - Kt[2]*valid_uv[0];
                    float a1 = Kt[1] - Kt[2]*valid_uv[1];
                    float b0 = KRKi_uv_homo[2]*valid_uv[0] - KRKi_uv_homo[0];
                    float b1 = KRKi_uv_homo[2]*valid_uv[1] - KRKi_uv_homo[1];
                    float inv_depth = (a0*b0 + a1*b1) / (a0*a0 + a1*a1);

                    pixel_point.inv_depths_.push_back(inv_depth);

                }
                
       
            }
            else
            {
                pixel_point.status_ = PixelPoint::Status::INVALID_COST_OBSERVATION;


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

 

bool MultiViewStereo::epipolarSearch(Image* const ref_image, uint32_t ref_u, uint32_t ref_v, const Image* const tar_image, 
     Eigen::Vector2f &tar_uv_min, Eigen::Vector2f &tar_uv_max)
{

    int ref_width = ref_image->getWidth();
    PixelPoint* const ref_ptr_pixel_point_matrix = ref_image->getPixelPointMatrixPtr();
    int tar_width = tar_image->getWidth();
    int tar_height = tar_image->getHeight();  
    float diagonal_length = sqrt(tar_width*tar_width + tar_height*tar_height);

    int ref_coord = ref_v * ref_width + ref_u;
    PixelPoint& pixel_point_in_ref_image = ref_ptr_pixel_point_matrix[ref_coord];


    Eigen::Vector2f epipolar_vector = Eigen::Vector2f(tar_uv_max[0] - tar_uv_min[0], tar_uv_max[1] - tar_uv_min[1]);
    float epipolar_length = epipolar_vector.norm();

	// ============== check the search distance. everything less than 3px is no need to refine (-> skip). ===================

    // if(epipolar_length <= 2){
    //     return false;
    // }
    Eigen::Vector2f unit_epipolar_vector =  epipolar_vector / epipolar_length;
    pixel_point_in_ref_image.unit_epipolar_vector_ = unit_epipolar_vector;

    DebugInfo* debug_info_ptr = &pixel_point_in_ref_image.debug_info_;
    if(config_->debug_plot_== true )
    {
        debug_info_ptr->unit_epipolar_vector_ = unit_epipolar_vector;
        debug_info_ptr->epipolar_length_ = epipolar_length;
    }

    // we don't want search too many thing
    if(std::abs(tar_uv_max[0] - tar_uv_min[0]) < 1e-3 || epipolar_length < 1e-3 || epipolar_length > diagonal_length/10)
    {
        return false;
    }

    bool tar_uv_min_isInImage = tar_image->isInImage(tar_uv_min[0], tar_uv_min[1], half_ws_);
    bool tar_uv_max_isInImage = tar_image->isInImage(tar_uv_max[0], tar_uv_max[1], half_ws_);
    if(tar_uv_min_isInImage == false || tar_uv_max_isInImage == false){
        return false;
    }


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
    int match_idx = 0;


    // for debug_plot
    float ws_2 = (2*half_ws_+1)*(2*half_ws_+1);
   

    while(s <= epipolar_length)
    {   
        Eigen::Vector2f tar_uv_current  = tar_uv_min + s * unit_epipolar_vector;
        bool isInImage = tar_image->isInImage(tar_uv_current[0], tar_uv_current[1], half_ws_);

        if(isInImage){
            // float cost = ZSAD(ref_ptr_pixel_point_matrix, ref_u, ref_v, tar_ptr_pixel_point_matrix, round(tar_uv_current[0]), round(tar_uv_current[1]), width, height, half_ws_)/(ws_2);
            float cost = SAD(ref_ptr_pixel_point_matrix, ref_u, ref_v, ref_width, tar_ptr_pixel_point_matrix, round(tar_uv_current[0]), round(tar_uv_current[1]), tar_width, half_ws_)/(ws_2); 
            // float cost = ASW(ref_ptr_pixel_point_matrix, ref_u, ref_v, ref_width, tar_ptr_pixel_point_matrix, round(tar_uv_current[0]), round(tar_uv_current[1]), tar_width, half_ws_); 
            pixel_point_in_ref_image.valid_uvs_.push_back(tar_uv_current);
            pixel_point_in_ref_image.costs_.push_back(cost);

            if(config_->debug_plot_ == true)
            {   
                debug_info_ptr->steps_.push_back(s);
            }
            
            if(cost < min_cost)
            {
                min_cost = cost;
                min_cost_idx = match_idx;
                // best_s_tmp = s;


            }

            match_idx+=1;
        }
        
        s+=1.0f;

    }

    int Ncost = pixel_point_in_ref_image.costs_.size();

    float left_max_cost = -1;
    int left_max_cost_idx = -1;
    float right_max_cost = -1;
    int right_max_cost_idx = -1;

    for(int cost_i = min_cost_idx - 1; cost_i >= 0; cost_i--)
    {
        float cost = pixel_point_in_ref_image.costs_[cost_i];
        if(cost > left_max_cost){
            left_max_cost = cost;
            left_max_cost_idx = cost_i;
        }

    }

    for(int cost_i = min_cost_idx + 1; cost_i < Ncost; cost_i++)
    {
        float cost = pixel_point_in_ref_image.costs_[cost_i];
        if(cost > right_max_cost){
            right_max_cost = cost;
            right_max_cost_idx = cost_i;
        }

    }

    if(ref_u == 472 && ref_v == 157)
    {
        std::cout << "debug" << std::endl;
    }


    if(min_cost <= ACCEPTABLE_MINI_COST && 
       min_cost_idx != -1 &&
       right_max_cost_idx != -1 &&
       left_max_cost_idx != -1 &&
       min_cost_idx != 0 &&  // make sure min_cost is not in the borther
       min_cost_idx != match_idx - 1 &&   // make sure min_cost is not in the borther
       left_max_cost - min_cost >= ACCEPTABLE_COST_DIFF && 
       right_max_cost - min_cost >= ACCEPTABLE_COST_DIFF 
       )
    {
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

    int height = ref_image->getHeight();
    int width = ref_image->getWidth();
    PixelPoint* ptr_pixel_point_matrix = ref_image->getPixelPointMatrixPtr();
	const int direction = is_forward ? 1 : -1;

    int direction_num = 0+(int)is_forward;

    for(int y = half_ws_; y < height - half_ws_; y++)
    {
        bool is_init_aggregate_path = false;
        int v = y;

        // forward aggregate
        for(int x = half_ws_; x < width - half_ws_; x++)
        {
            int u = (is_forward) ? x : width - x - 1;

            if(v < config_->start_match_uv_[1] || u < config_->start_match_uv_[0]){
                continue;
            }

            if((u == 752 && v == 330))
            {
                std::cout << "debug" << std::endl;
            }

            int current_coord = v*width + u;
            PixelPoint& pixel_point = ptr_pixel_point_matrix[current_coord];

            if(pixel_point.status_ == PixelPoint::Status::INVALID_COST_OBSERVATION || pixel_point.status_ == PixelPoint::Status::DISABLE){
                is_init_aggregate_path = false;
                continue;
            }


            // init aggregate path
            if(is_init_aggregate_path == false)
            {
                init_aggregate_pixel(pixel_point, direction_num);
                is_init_aggregate_path = true;
                continue;
            }

            // forward update
            PixelPoint& pixel_point_in_last_pos = ptr_pixel_point_matrix[current_coord-direction];

            cost_aggregation_from_last_pixel(pixel_point_in_last_pos, pixel_point, direction_num);

            // if(cost_aggregation_from_last_pixel(pixel_point_in_last_pos, pixel_point, direction_num) == true)
            // {

            // }
            // // fail in aggregation, then just copy cost to tmp aggregate cost
            // else{

            //     init_aggregate_pixel(pixel_point, direction_num);

            // }


        }

    }

}

void MultiViewStereo::cost_aggregation_up_down(Image* const ref_image, bool is_forward)
{

    int height = ref_image->getHeight();
    int width = ref_image->getWidth();
    PixelPoint* ptr_pixel_point_matrix = ref_image->getPixelPointMatrixPtr();
	const int direction = is_forward ? 1 : -1;

    int direction_num = 2+(int)is_forward;

    for(int x = half_ws_; x < (int)width - half_ws_; x++)
    {
        bool is_init_aggregate_path = false;
        int u = x;

        // forward aggregate
        for(int y = half_ws_; y < (int)height - half_ws_; y++)
        {
            int v = (is_forward) ? y : height - y - 1;

            if(v < config_->start_match_uv_[1] || u < config_->start_match_uv_[0]){
                continue;
            }

            if((u == 752 && v == 330))
            {
                std::cout << "debug" << std::endl;
            }

            int current_coord = v*width + u;
            PixelPoint& pixel_point = ptr_pixel_point_matrix[current_coord];

            if(pixel_point.status_ == PixelPoint::Status::INVALID_COST_OBSERVATION || pixel_point.status_ == PixelPoint::Status::DISABLE){
                is_init_aggregate_path = false;
                continue;
            }


            // init aggregate path
            if(is_init_aggregate_path == false)
            {
                init_aggregate_pixel(pixel_point, direction_num);
                is_init_aggregate_path = true;
                continue;
            }

            // forward update
            PixelPoint& pixel_point_in_last_pos = ptr_pixel_point_matrix[current_coord-direction*width];

            cost_aggregation_from_last_pixel(pixel_point_in_last_pos, pixel_point, direction_num);
            
            // if(cost_aggregation_from_last_pixel(pixel_point_in_last_pos, pixel_point, direction_num) == true)
            // {

            // }
            // // fail in aggregation, then just copy cost to tmp aggregate cost
            // else{
            //     init_aggregate_pixel(pixel_point, direction_num);

            // }

        }

    }

}



void MultiViewStereo::cost_aggregation_from_last_pixel(PixelPoint& pixel_point_in_last_pos, PixelPoint& pixel_point, int direction_num)
{

    int minimum_cost_idx_in_curr_pos = std::min_element(pixel_point.costs_.begin(), pixel_point.costs_.end()) - pixel_point.costs_.begin();
    float mini_cost_in_curr_pos = pixel_point.costs_[minimum_cost_idx_in_curr_pos];
                    
    float P = MAXIMUM_AGGREAGTE_COST_PENALTY*mini_cost_in_curr_pos/(std::abs(pixel_point.intensity_ - pixel_point_in_last_pos.intensity_) + 1);


    if(pixel_point_in_last_pos.status_ == PixelPoint::Status::NORMAL)
    {

        int minimum_cost_idx_in_last_pos = std::min_element(pixel_point_in_last_pos.tmp_aggregate_costs_.begin(), pixel_point_in_last_pos.tmp_aggregate_costs_.end()) - pixel_point_in_last_pos.tmp_aggregate_costs_.begin();
        float mini_tmp_aggregated_cost_in_last_pos = pixel_point_in_last_pos.tmp_aggregate_costs_[minimum_cost_idx_in_last_pos];

        float inv_depth_in_last_pos_in_at_mini_cost = pixel_point_in_last_pos.inv_depths_[minimum_cost_idx_in_last_pos];

        float depth_in_last_pos_in_at_mini_cost = 1.0f/inv_depth_in_last_pos_in_at_mini_cost;
        float acceptable_depth = depth_in_last_pos_in_at_mini_cost * ACCEPTABLE_DEPTH_PARAMETER;
        float depth_minus_in_last_pos_in_at_mini_cost = depth_in_last_pos_in_at_mini_cost - acceptable_depth;
        float inv_depth_minus_in_last_pos_in_at_mini_cost = 1/depth_minus_in_last_pos_in_at_mini_cost;
        float depth_plus_in_last_pos_in_at_mini_cost = depth_in_last_pos_in_at_mini_cost + acceptable_depth;
        float inv_depth_plus_in_last_pos_in_at_mini_cost = 1/depth_plus_in_last_pos_in_at_mini_cost;

        for(int cost_idx = 0; cost_idx < (int)pixel_point.costs_.size(); cost_idx++)
        {

            float inv_depth = pixel_point.inv_depths_[cost_idx];
            float depth = 1/inv_depth;
            float tmp_aggregated_cost_in_last_pos = mini_tmp_aggregated_cost_in_last_pos;

            // if(cost_idx == 1)
            // {
            //     std::cout << "debug" << std::endl;
            // }

            if(depth < depth_minus_in_last_pos_in_at_mini_cost){


                tmp_aggregated_cost_in_last_pos = pixel_point_in_last_pos.getInterpolatedTmpAggregatedCostByInvDepth(1.0f/(depth+acceptable_depth));

            }
            else if(depth > depth_plus_in_last_pos_in_at_mini_cost)
            {

                tmp_aggregated_cost_in_last_pos = pixel_point_in_last_pos.getInterpolatedTmpAggregatedCostByInvDepth(1.0f/(depth-acceptable_depth));

            }

            // tmp_aggregated_cost_in_last_pos = pixel_point_in_last_pos.getInterpolatedTmpAggregatedCostByInvDepth(inv_depth);



            float depth_in_curr_pos = 1/inv_depth;

            float most_likely_depth_in_last_pos = 1/inv_depth_in_last_pos_in_at_mini_cost;


            float tmp_aggregate_cost = std::min( 
                tmp_aggregated_cost_in_last_pos - mini_tmp_aggregated_cost_in_last_pos, P
            );

                
            pixel_point.tmp_aggregate_costs_[cost_idx] = pixel_point.costs_[cost_idx] + tmp_aggregate_cost;
            pixel_point.aggregate_costs_[cost_idx] += pixel_point.tmp_aggregate_costs_[cost_idx]; 

            if(config_->debug_plot_ == true)
            {
                pixel_point.debug_info_.appendTmpAggregateCostsFromCertainDirection(tmp_aggregate_cost, direction_num);
            
            }
        }

    }
    else
    {
        
        float acceptable_depth_minus = pixel_point_in_last_pos.min_depth_ * ACCEPTABLE_DEPTH_PARAMETER;
        float acceptable_depth_plus = pixel_point_in_last_pos.min_depth_ * ACCEPTABLE_DEPTH_PARAMETER;


        for(int cost_idx = 0; cost_idx < (int)pixel_point.costs_.size(); cost_idx++)
        {
            float depth = 1/pixel_point.inv_depths_[cost_idx];
            float tmp_aggregate_cost = 0;
            if(depth < acceptable_depth_minus || depth > acceptable_depth_plus)
            {
                tmp_aggregate_cost = P;
            }

            pixel_point.tmp_aggregate_costs_[cost_idx] = pixel_point.costs_[cost_idx] + tmp_aggregate_cost;
            pixel_point.aggregate_costs_[cost_idx] += pixel_point.tmp_aggregate_costs_[cost_idx]; 

            if(config_->debug_plot_ == true)
            {
                pixel_point.debug_info_.appendTmpAggregateCostsFromCertainDirection(tmp_aggregate_cost, direction_num);
            
            }

        }

    }

    pixel_point.num_has_been_aggregated_++;



   
}




void MultiViewStereo::init_aggregate_pixel(PixelPoint& pixel_point, int direction_num)
{
    pixel_point.tmp_aggregate_costs_ = pixel_point.costs_;
    for(int cost_idx=0; cost_idx < pixel_point.aggregate_costs_.size(); cost_idx++)
    {
        pixel_point.aggregate_costs_[cost_idx] += pixel_point.tmp_aggregate_costs_[cost_idx]; 
    }

    pixel_point.num_has_been_aggregated_++;

    if(config_->debug_plot_ == true)
    {
        pixel_point.debug_info_.setZerosTmpAggregateCostsFromCertainDirection(pixel_point.costs_.size(), direction_num);
    }

}


void MultiViewStereo::consistency_check()
{
    // left right consistency
    std::cout << "start to point cloud consistency check" << std::endl;
    std::vector<Image*>& images = dataset_->getImages();
    for(int ref_idx=0; ref_idx < images.size(); ref_idx++)
    {
        Image* ref_image = images[ref_idx];
        int height = ref_image->getHeight();
        int width = ref_image->getWidth();
        std::map<int, Image*>& map_of_images_has_been_matched = ref_image->getMapOfImagesHasBeenMatched();
        PixelPoint* ptr_pixel_point_matrix = ref_image->getPixelPointMatrixPtr();
        Eigen::Matrix4f T_world_ref = ref_image->getTransformationMatrix();
        Eigen::Matrix4f T_ref_world = invertTransform(T_world_ref);
        Eigen::Matrix3f K_ref = config_->cameras_[ref_image->getCameraId()].getIntrinsicsMatrix();

        for(int v=half_ws_; v < height - half_ws_; v++)
        {
            for(int u=half_ws_; u < width - half_ws_; u++)
            {

                if(v < config_->start_match_uv_[1] || u < config_->start_match_uv_[0]){
                    continue;
                }

                // std::cout << "debug " << " u= " << u << ", v= " << v << std::endl;

                int current_coord = v*width + u;

                PixelPoint& ref_ptr_pixel_point = ptr_pixel_point_matrix[current_coord];
                float z = ref_ptr_pixel_point.output_data_.depth_;
                float min_z = ref_ptr_pixel_point.min_depth_;
                float max_z = ref_ptr_pixel_point.max_depth_;

                if(max_z - min_z > 0.5f || z < config_->min_depth_ || z > config_->max_depth_)
                {
                    continue;
                }

                // std::cout << "phase 1 in coord: (" << u << "," << v << ")" << std::endl;

                float x = (u - K_ref(0,2)) * z / K_ref(0,0);
                float y = (v - K_ref(1,2)) * z / K_ref(1,1);

                // const cv::Vec3b& pix_bgr = ref_cv_bgr_data.at<cv::Vec3b>(v, u);
                Eigen::Vector4f pref_h = Eigen::Vector4f(x,y,z, 1.0f); 
                Eigen::Vector4f pw_h = T_world_ref * pref_h;
                // Eigen::Vector3f pw = pw_h.head<3>() / pw_h[3]; 
                ref_ptr_pixel_point.num_correct_consistency_check_ = 0;

                for (const auto& [id, tar_image] : map_of_images_has_been_matched) 
                {
                    // std::cout << "checking between " << ref_image->getId() << " and " << tar_image->getId() << std::endl;
                    Eigen::Matrix4f T_world_tar = tar_image->getTransformationMatrix();
                    Eigen::Matrix4f T_tar_world = invertTransform(T_world_tar);
                    Eigen::Matrix3f K_tar = config_->cameras_[tar_image->getCameraId()].getIntrinsicsMatrix();

                    Eigen::Vector4f ptar_h = T_tar_world * pw_h;
                    Eigen::Vector3f ptar = ptar_h.head<3>() / ptar_h[3]; 

                    // std::cout <<"phase 2:" << "ptar[2]=" << ptar[2] <<" in coord: (" << u << "," << v << ")" << std::endl;

                    if(ptar[2] <= 0)
                    {
                        continue;
                    }


                    PixelPoint* ptr_pixel_point_matrix_in_tar_image = tar_image->getPixelPointMatrixPtr();

                    int height_in_tar_image = tar_image->getHeight();
                    int width_in_tar_image = tar_image->getWidth();

                    int u_in_tar = round(ptar[0]/ptar[2]*K_tar(0,0) + K_tar(0,2));
                    int v_in_tar = round(ptar[1]/ptar[2]*K_tar(1,1) + K_tar(1,2));

                    bool isInImage = false;
                    isInImage = tar_image->isInImage(u_in_tar, v_in_tar, half_ws_);

                    // std::cout << "phase 3 in coord: " << " is in image " << isInImage << "(" << u << "," << v << ")" << std::endl;

                    if(isInImage == false)
                    {
                        continue;
                    }



                    int current_coord_in_tar_image = v_in_tar*width_in_tar_image + u_in_tar;

                    PixelPoint& tar_ptr_pixel_point = ptr_pixel_point_matrix_in_tar_image[current_coord_in_tar_image];
                    float z_in_tar = tar_ptr_pixel_point.output_data_.depth_;
                    float min_z_in_tar = tar_ptr_pixel_point.min_depth_;
                    float max_z_in_tar = tar_ptr_pixel_point.max_depth_;

                    // std::cout << "phase 4 " << "max_z_in_tar : " << max_z_in_tar << " min_z_in_tar :" <<  min_z_in_tar << " in coord: (" << u << "," << v << ") for target image ID :" << tar_image->getId()  << std::endl;

                    if(max_z_in_tar - min_z_in_tar > 0.5f || z_in_tar < config_->min_depth_ || z_in_tar > config_->max_depth_)
                    {
                        continue;
                    }


                    float x_in_tar = (u_in_tar - K_tar(0,2)) * z_in_tar / K_tar(0,0);
                    float y_in_tar = (v_in_tar - K_tar(1,2)) * z_in_tar / K_tar(1,1);

                    // const cv::Vec3b& pix_bgr = ref_cv_bgr_data.at<cv::Vec3b>(v, u);
                    Eigen::Vector4f reprojected_ptar_h = Eigen::Vector4f(x_in_tar,y_in_tar,z_in_tar, 1.0f); 
                    Eigen::Vector4f reprojected_pw_h = T_world_tar * reprojected_ptar_h;
                    Eigen::Vector4f reprojected_pref_h = T_ref_world * reprojected_pw_h;
                    Eigen::Vector3f reprojected_pref = reprojected_pref_h.head<3>() / reprojected_pref_h[3]; 

                    // std::cout << "phase 5 " << "reprojected_pref[2] : " << reprojected_pref[2] << " in coord: (" << u << "," << v << ")" << std::endl;

                    if(reprojected_pref[2] <= 0)
                    {
                        continue;
                    }



                    float reprojected_u = reprojected_pref[0]/reprojected_pref[2]*K_ref(0,0) + K_ref(0,2);
                    float reprojected_v = reprojected_pref[1]/reprojected_pref[2]*K_ref(1,1) + K_ref(1,2);
                    
                    isInImage = ref_image->isInImage(reprojected_u, reprojected_v, half_ws_);
                    // std::cout << "phase 6 in coord: " << " is in image " << isInImage << "(" << u << "," << v << ")" << std::endl;

                    if(isInImage == false)
                    {
                        continue;
                    }


                    float u_diff = u - reprojected_u;
                    float v_diff = v - reprojected_v;
                    float reprojected_error = sqrt(u_diff*u_diff+v_diff*v_diff);
                    // std::cout << "consistency check reprojected error: " << reprojected_error << " in coord: (" << u << "," << v << ")" << std::endl;

                    // sqrt(1^2+1^2), maximum 1 pixel error
                    if(reprojected_error <= 1.4142)
                    {
                        ref_ptr_pixel_point.num_correct_consistency_check_++;
                        // std::cout << "consistency check pass with reprojected error: " << reprojected_error << " for image id " <<  ref_image->getId() << " in coord: (" << u << "," << v << ")" << std::endl;
                    }

                }

            }
        }


        
    }


    std::cout << "point cloud consistency check end" << std::endl;

}

void MultiViewStereo::update_uncertainty(Image* const ref_image)
{

    float const_error_in_pixel = 1.0f;

    Eigen::Matrix3f KRKi = ref_image->getKRKi();
    Eigen::Vector3f Kt = ref_image->getKt();

    int height = ref_image->getHeight();
    int width = ref_image->getWidth();

    PixelPoint* ptr_pixel_point_matrix = ref_image->getPixelPointMatrixPtr();


    for(int v=half_ws_; v < height - half_ws_; v++)
    {
        for(int u=half_ws_; u < width - half_ws_; u++)
        {

            if(v < config_->start_match_uv_[1] || u < config_->start_match_uv_[0]){
                continue;
            }

            if(u == 170 && v == 73){
                std::cout <<"debug" << std::endl;
            }
            // std::cout << "debug " << " u= " << u << ", v= " << v << std::endl;

            int current_coord = v*width + u;

            PixelPoint& pixel_point = ptr_pixel_point_matrix[current_coord];

            if(pixel_point.status_ == PixelPoint::Status::INVALID_COST_OBSERVATION || pixel_point.status_ == PixelPoint::Status::DISABLE){
                continue;
            }
                
            DebugInfo* debug_info_ptr = nullptr;
            if(config_->debug_plot_ == true)
            {
                debug_info_ptr = &pixel_point.debug_info_;
            }

            Eigen::Vector3f KRKi_uv_homo = KRKi * Eigen::Vector3f(u,v,1);

            if(pixel_point.num_has_been_aggregated_ == 0)
            {
                pixel_point.num_has_been_aggregated_ = 1;
                pixel_point.aggregate_costs_ = pixel_point.costs_;
            }

            
            int mini_aggregate_cost_idx = std::min_element(pixel_point.aggregate_costs_.begin(), pixel_point.aggregate_costs_.end()) - pixel_point.aggregate_costs_.begin();
            float min_aggregate_cost = pixel_point.aggregate_costs_[mini_aggregate_cost_idx];

            // for(float &aggregate_cost : pixel_point.aggregate_costs_)
            // {
            //     aggregate_cost -= min_aggregate_cost;
            // }
            // min_aggregate_cost = 0;
            
            if(mini_aggregate_cost_idx == 0 || mini_aggregate_cost_idx == pixel_point.aggregate_costs_.size() - 1)
            {
                pixel_point.status_ = PixelPoint::Status::INVALID_AGGREGATION_COST_OBSERVATION;
                continue;
            }

            float mini_quadratic_aggregate_cost_idx = (float)mini_aggregate_cost_idx;
            int mini_aggregate_cost_idx_plus_1 = mini_aggregate_cost_idx + 1;
            int mini_aggregate_cost_idx_minus_1 = mini_aggregate_cost_idx - 1;



            // when there are at least one aggregated cost point in plus and minus direction, 
            // depth uncertainty will be considered to reduce

            
            float min_aggregate_cost_plus_1 = pixel_point.aggregate_costs_[mini_aggregate_cost_idx_plus_1];
            float min_aggregate_cost_minus_1 = pixel_point.aggregate_costs_[mini_aggregate_cost_idx_minus_1];
            float alpha2 = min_aggregate_cost_minus_1 + min_aggregate_cost_plus_1 - 2 * min_aggregate_cost;


            float minus_beta2 = min_aggregate_cost_minus_1 - min_aggregate_cost_plus_1;
            float beta_square = minus_beta2 * minus_beta2;
            mini_quadratic_aggregate_cost_idx = (float)mini_aggregate_cost_idx + (minus_beta2)/ (2.0f * alpha2);
            min_aggregate_cost = min_aggregate_cost - (minus_beta2) * (minus_beta2) / (8.0f* alpha2);

            float acceptable_margin = pixel_point.num_has_been_aggregated_ * ACCEPTABLE_COST_DIFF;
            float acceptable_threshold = min_aggregate_cost + acceptable_margin;
            
            float root_equation = sqrt(beta_square + 2 * alpha2 * acceptable_margin);

            float mini_aggregate_cost_idx_plus = mini_aggregate_cost_idx + 0.5;

            if(min_aggregate_cost_plus_1 >= acceptable_threshold)
            {
                mini_aggregate_cost_idx_plus = std::max(mini_aggregate_cost_idx_plus, mini_quadratic_aggregate_cost_idx + (minus_beta2/2 + root_equation)/alpha2);
            }
            else
            {
                for(int cost_i = mini_aggregate_cost_idx_plus_1 + 1; cost_i < pixel_point.aggregate_costs_.size(); cost_i++)
                {
                    float curr_cost = pixel_point.aggregate_costs_[cost_i];
                    float last_cost = pixel_point.aggregate_costs_[cost_i-1];

                    if(curr_cost > acceptable_threshold && last_cost <= acceptable_threshold)
                    {
                        mini_aggregate_cost_idx_plus = (float)(cost_i-1) + (acceptable_threshold - last_cost)/(curr_cost - last_cost);
                        break;
                    }
                }

            }


            float mini_aggregate_cost_idx_minus = mini_aggregate_cost_idx - 0.5;

            if(min_aggregate_cost_minus_1 >= acceptable_threshold)
            {
                mini_aggregate_cost_idx_minus = std::min(mini_aggregate_cost_idx_plus, mini_quadratic_aggregate_cost_idx + (minus_beta2/2 - root_equation)/alpha2);

            }
            else
            {
                for(int cost_i = mini_aggregate_cost_idx_minus_1 - 1; cost_i >= 0; cost_i--)
                {

                    float curr_cost = pixel_point.aggregate_costs_[cost_i];
                    float last_cost = pixel_point.aggregate_costs_[cost_i+1];

                    if(curr_cost > acceptable_threshold && last_cost <= acceptable_threshold)
                    {
                        mini_aggregate_cost_idx_minus = (float)(cost_i+1) - (acceptable_threshold - last_cost)/(curr_cost - last_cost);
                        break;
                    }
                }
            }


            bool isUniquenessUncertaintyRegion = true;
            for(int cost_i = 0; cost_i < pixel_point.aggregate_costs_.size(); cost_i++)
            {
                if( cost_i >= mini_aggregate_cost_idx_minus && cost_i <= mini_aggregate_cost_idx_plus)
                {
                    continue;
                }
                
                float curr_cost = pixel_point.aggregate_costs_[cost_i];

                if(curr_cost <=  acceptable_threshold)
                {
                    //reject this uncertainty reduce
                    isUniquenessUncertaintyRegion = false;
                    break;
                }

            }
            
            Eigen::Vector2f uv_possible_match = pixel_point.valid_uvs_[0] + mini_quadratic_aggregate_cost_idx * pixel_point.unit_epipolar_vector_;
            Eigen::Vector2f uv_possible_match_minus = pixel_point.valid_uvs_[0] + mini_aggregate_cost_idx_minus * pixel_point.unit_epipolar_vector_;
            Eigen::Vector2f uv_possible_match_plus = pixel_point.valid_uvs_[0] + mini_aggregate_cost_idx_plus * pixel_point.unit_epipolar_vector_;


            if(config_->debug_plot_ == true)
            {

                debug_info_ptr->mini_aggregate_cost_idx_ = mini_aggregate_cost_idx;
                debug_info_ptr->mini_aggregate_cost_idx_minus_ = floor(mini_aggregate_cost_idx_minus);
                debug_info_ptr->mini_aggregate_cost_idx_plus_ = ceil(mini_aggregate_cost_idx_plus);
                debug_info_ptr->uv_possible_match_ = uv_possible_match;

            }

            if(isUniquenessUncertaintyRegion == false)
            {
                pixel_point.status_ = PixelPoint::Status::INVALID_NON_UNIQUENESSS;
                continue;
            }




            // Eigen::Vector2f uv_possible_match_minus = uv_possible_match - const_error_in_pixel*pixel_point.unit_epipolar_vector_;
            // Eigen::Vector2f uv_possible_match_plus = uv_possible_match + const_error_in_pixel*pixel_point.unit_epipolar_vector_;

            float min_inv_depth, max_inv_depth = -1.0f;
            pixel_depth_estimation(min_inv_depth, max_inv_depth, Kt, KRKi_uv_homo, uv_possible_match_minus, uv_possible_match_plus);
            pixel_point.setMaxInvDepth(max_inv_depth);
            pixel_point.setMinInvDepth(min_inv_depth);
            pixel_point.output_data_.depth_ = (pixel_point.min_depth_ + pixel_point.max_depth_) / 2.0f;

    

            if(config_->debug_plot_ == true){

                debug_info_ptr->updated_min_depth_ = pixel_point.min_depth_;
                debug_info_ptr->updated_max_depth_ = pixel_point.max_depth_;
                debug_info_ptr->updated_depth_ = pixel_point.output_data_.depth_;

                // std::cout << "min_depth : " << min_depth << " and max_depth : " << max_depth << " at uv_best_match (" << uv_best_match[0] << "," << uv_best_match[1] << ")" << std::endl;

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



    
}