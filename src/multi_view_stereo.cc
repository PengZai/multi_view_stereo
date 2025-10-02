#include "multi_view_stereo.h"

namespace MVS
{


MultiViewStereo::MultiViewStereo(Config* const config)
{
    config_ = config;
    ws_ = config_->window_size_;
    max_inv_depth_ = 1/config->min_depth_;
    min_inv_depth_ = 1/config->max_depth_;


}

MultiViewStereo::~MultiViewStereo()
{
}


void MultiViewStereo::setDataset(Dataset* const dataset)
{
    dataset_ = dataset;
}


void MultiViewStereo::setReferenceImage(Image* const image)
{
    ref_image_ = image;
}
 

void MultiViewStereo::run()
{

    Image* ref_image = dataset_->images_[config_->ref_image_idx_];
    setReferenceImage(ref_image);

    for(size_t i=config_->tar_image_start_idx_;i < dataset_->images_.size(); i++)
    {
        Image* image = dataset_->images_[i];
        if(ref_image_->getId() == image->getId()){
            continue;
        }

        auto start = std::chrono::steady_clock::now();        
        match(ref_image_, image, config_->debug_plot_);
        auto end = std::chrono::steady_clock::now();
        auto tt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("match Done! Timing : %lf s for ref %d and tar %d\n", tt.count() / 1000.0, config_->ref_image_idx_, config_->tar_image_start_idx_);

        break;
    }

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

    cv::Mat ref_gray_data = ref_image->getGrayData();
    cv::Mat ref_rgb_data = ref_image->getRGBData();



    for(size_t v=ws_; v < ref_gray_data.rows - ws_; v++)
    {
        for(size_t u=ws_; u < ref_gray_data.cols - ws_; u++)
        {
            // KRKi * (u,v,1) + dmin_inv * Kt = pt_min
            // KRKi * (u,v,1) + dmax_inv * Kt = pt_max
            // u_min, v_min = pt_min / pt_min[2]
            // u_max, v_max = pt_max / pt_max[2]

            if(v < config_->start_match_uv_[1] || u < config_->start_match_uv_[0]){
                continue;
            }

            // std::cout << "epipolar search for " << u << ", "<< v << std::endl;

            if(debug_plot)
            {
                cv::Mat vis_ref_rgb_data = ref_rgb_data.clone();
                cv::circle(vis_ref_rgb_data, cv::Point2i(u,v), 3, cv::Scalar(0,0,255), 2);
                cv::putText(vis_ref_rgb_data, "("+std::to_string(u)+","+std::to_string(v)+")",cv::Point2i(u,v), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,0,255), 1.8);

                cv::imwrite(config_->save_figure_path_ + "/vis_ref_rgb_data.png", vis_ref_rgb_data);
                cv::imshow("vis_ref_rgb_data", vis_ref_rgb_data);

            }
        
            // auto start = std::chrono::steady_clock::now();        
            Eigen::Vector3f KRKi_uv_homo = KRKi * Eigen::Vector3f(u,v,1);
            Eigen::Vector3f pt_min = KRKi_uv_homo + max_inv_depth_ * Kt;
            Eigen::Vector3f pt_max = KRKi_uv_homo + min_inv_depth_ * Kt;
            Eigen::Vector2f uv_min = Eigen::Vector2f(pt_min[0]/pt_min[2], pt_min[1]/pt_min[2]);
            Eigen::Vector2f uv_max = Eigen::Vector2f(pt_max[0]/pt_max[2], pt_max[1]/pt_max[2]);
            Eigen::Vector2f uv_best_match;
            // auto end = std::chrono::steady_clock::now();    
            // auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            // printf("epipolar line generation ! Timing : %lld µs\n", (long long)elapsed_us);

            // start = std::chrono::steady_clock::now();        
            // const cv::Mat ref_gray_patch = getSubpixelPatch(ref_gray_data, u,  v, ws_, ws_);
            cv::Mat ref_gray_patch;
            getRoundPixelPatch(ref_gray_data, u, v, ws_, ws_, ref_gray_patch);
            // end = std::chrono::steady_clock::now();
            // elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            // printf("getSubpixelPatch done ! Timing : %lld µs\n", (long long)elapsed_us);

            // auto start = std::chrono::steady_clock::now();        
            epipolarSearch(ref_gray_patch, tar_image, uv_min, uv_max, uv_best_match, debug_plot);
            // auto end = std::chrono::steady_clock::now();
            // auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            // printf("epipolar search done ! Timing : %lld µs\n", (long long)elapsed_us);


        }

    }

    if(debug_plot)
    {
    }

}

void MultiViewStereo::epipolarSearch(const cv::Mat ref_data_patch, const Image* const tar_image, 
    const Eigen::Vector2f &uv_min, const Eigen::Vector2f &uv_max, Eigen::Vector2f uv_best_match, bool debug_plot)
{

    cv::Mat tar_data = tar_image->getGrayData();
    Eigen::Vector2f epipolar_vector = Eigen::Vector2f(uv_max[0] - uv_min[0], uv_max[1] - uv_min[1]);
    float epipolar_length = epipolar_vector.norm();
    epipolar_vector.normalize();

    float s = 0;
    float min_cost = std::numeric_limits<float>::infinity();
    int best_s_idx = 0;
    Eigen::Vector2f uv_best_tmp(-1, -1);
    int best_s_idx_temp = 0;
    std::vector<float> costs;
    std::vector<float> steps;
    std::vector<Eigen::Vector2f> valid_uvs;
    while(s <= epipolar_length)
    {   
        Eigen::Vector2f uv_current  = uv_min + s * epipolar_vector;
        if(tar_image->isInImage(uv_current[0], uv_current[1], ws_/2)){
            cv::Mat tar_data_patch;
            // cv::getRectSubPix(tar_data, cv::Size(ws_, ws_), cv::Point2f(uv_current[0],  uv_current[1]), tar_data_patch, CV_32F);
            getRoundPixelPatch(tar_data, uv_current[0], uv_current[1], ws_, ws_, tar_data_patch);
            float cost = SAD(ref_data_patch, tar_data_patch);
            valid_uvs.push_back(uv_current);
            costs.push_back(cost);
            steps.push_back(s);
            if(cost < min_cost)
            {
                min_cost = cost;
                uv_best_tmp = uv_current;
                best_s_idx_temp = steps.size()-1;
            }
        }
        

        s+=1.0;

    }

    // std::cout << "epipolar_length : " << steps.size() << std::endl;

    if(min_cost < std::numeric_limits<float>::infinity()){
        uv_best_match = uv_best_tmp;
        best_s_idx = best_s_idx_temp;

        if(debug_plot == true)
        {

            cv::Point2i cv_uv_best(round(uv_best_tmp[0]),round(uv_best_tmp[1]));
            cv::Point2i cv_uv_start(round(valid_uvs.front()[0]),round(valid_uvs.front()[1]));
            cv::Point2i cv_uv_end(round(valid_uvs.back()[0]),round(valid_uvs.back()[1]));

            matplot::plot(steps, costs, "-o");   // "-o" = line with circle markers

            matplot::xlabel("Step");
            matplot::ylabel("Cost");
            matplot::title("best at  " + std::to_string(cv_uv_best.x) + "," + std::to_string(cv_uv_best.y) + 
            ", from (" + std::to_string(cv_uv_start.x)+","+std::to_string(cv_uv_start.y) + ") to (" + 
            std::to_string(cv_uv_end.x)+","+std::to_string(cv_uv_end.y)+")");

            matplot::text(steps[best_s_idx], costs[best_s_idx], "best");
            matplot::save(config_->save_figure_path_ + "/cost_vs_step.png");

            cv::Mat vis_tar_data = tar_image->getRGBData().clone();



            std::cout << "uv_best_tmp : " << uv_best_tmp[0] << " , " << uv_best_tmp[1] << " starting from (" 
                << cv_uv_start.x << "," << cv_uv_start.y  << ") end at (" << cv_uv_end.x << "," << cv_uv_end.y << ")" << std::endl;
            cv::circle(vis_tar_data, cv_uv_best, 3, cv::Scalar(255,0,255), 2); // pink
            cv::circle(vis_tar_data, cv_uv_start, 2, cv::Scalar(0,0,255), 2); // red
            cv::putText(vis_tar_data, "S",cv_uv_start, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 1.8);
            cv::line(vis_tar_data, cv_uv_start, cv_uv_end, cv::Scalar(0,255,0), 1); // green
            cv::circle(vis_tar_data, cv_uv_end, 2, cv::Scalar(255,0,0), 2); // blue
            cv::putText(vis_tar_data, "E",cv_uv_end, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,0,0), 1.8);
            cv::imwrite(config_->save_figure_path_ + "/vis_tar_data.png", vis_tar_data);
            cv::imshow("vis_tar_data", vis_tar_data);
            cv::waitKey(0);

            


        }
    }
    else
    {
        // std::cout << " nothing to match " << std::endl;
    }

    

    

}


float MultiViewStereo::SAD(const cv::Mat p1, const cv::Mat p2)
{
    CV_Assert(p1.size() == p2.size());

    float cost = 0;
    const int rows = p1.rows;
    const int cols = p1.cols;

    for (int y = 0; y < rows; y++)
        for (int x = 0; x < cols; x++)
            cost += std::abs(p1.at<uchar>(y,x) - p2.at<uchar>(y,x));
    return cost;

    // return static_cast<float>(cv::norm(p1, p2, cv::NORM_L1));


    // cv::Mat diff;
    // cv::absdiff(p1, p2, diff);   // element-wise |p1 - p2|
    // float cost = static_cast<float>(cv::sum(diff)[0]);  // sum over all pixels
    // return cost;
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