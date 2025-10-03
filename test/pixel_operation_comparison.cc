#include <iostream>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>


template<typename Func>
void measureTime(Func f){


    auto start = std::chrono::steady_clock::now();        
    f();
    auto end = std::chrono::steady_clock::now();    
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double elapsed_s = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    std::cout << "Elapsed: " << elapsed_us << " us " << "and " << elapsed_s << " s" <<"\n";

}


float SADPatch_with_AT(const cv::Mat& ref_data, const cv::Mat& tar_data, int u, int v, int half_ws) {

    float cost = 0.0f;

    // Patch loop
    for (int dy = -half_ws; dy <= half_ws; dy++) {
        for (int dx = -half_ws; dx <= half_ws; dx++) {
            cost += std::abs(ref_data.at<uint8_t>(v, u + dx) - tar_data.at<uint8_t>(v, u+dx));
        }
    }
    return cost;
}

void cv_at_with_sad(const cv::Mat& ref_data, const cv::Mat tar_data, const uint32_t width, const uint32_t height, const uint32_t max_step)
{
    int half_ws = 2;
    float total_cost = 0;
    std::vector<float> costs;
    for (int v = half_ws; v < height-half_ws; v++) {
        for (int u = half_ws; u < width - half_ws; u++) {
            for(uint32_t s=0; s < max_step; s++){

                int current_u = u+s;
                int current_v = v;
                if(current_u >=half_ws && current_v >= half_ws && current_u < width - half_ws - 1 && current_v < height - half_ws - 1){
                    
                    float cost = SADPatch_with_AT(ref_data, tar_data, current_u, current_v, half_ws);
                    total_cost += cost;
                    // costs.push_back(cost);
                }
            }
            // std::cout <<"(u,v) " << u << "," << v << "costs:" << costs << std::endl;
        }
    }
    std::cout << "total cost:" << total_cost << " size of costs " << costs.size() << std::endl;

}


float SADPatch_with_cvptr(const cv::Mat& ref_data, const cv::Mat& tar_data, int u, int v, int half_ws) {

    float cost = 0.0f;

    // Patch loop
    for (int dy = -half_ws; dy <= half_ws; dy++) {

        const uint8_t* ref_row = ref_data.ptr<uint8_t>(v + dy);  // pointer to row in ref
        const uint8_t* tar_row = tar_data.ptr<uint8_t>(v + dy);  // pointer to row in tar

        for (int dx = -half_ws; dx <= half_ws; dx++) {
            cost += std::abs(*(ref_row + u + dx) - *(tar_row + u + dx));
        }
    }
    return cost;
}

void cv_ptr_with_sad(const cv::Mat& ref_data, const cv::Mat tar_data, const uint32_t width, const uint32_t height, const uint32_t max_step)
{
    int half_ws = 2;
    float total_cost = 0;
    std::vector<float> costs;
    for (int v = half_ws; v < height-half_ws; v++) {
        for (int u = half_ws; u < width - half_ws; u++) {
            for(uint32_t s=0; s < max_step; s++){

                int current_u = u+s;
                int current_v = v;
                if(current_u >=half_ws && current_v >= half_ws && current_u < width - half_ws - 1 && current_v < height - half_ws - 1){
                    
                    float cost = SADPatch_with_cvptr(ref_data, tar_data, current_u, current_v, half_ws);
                    total_cost += cost;
                    // costs.push_back(cost);
                }
            }
            // std::cout <<"(u,v) " << u << "," << v << "costs:" << costs << std::endl;
        }
    }
    std::cout << "total cost:" << total_cost << " size of costs " << costs.size() << std::endl;

}


float SADPatch_with_ptr(const uint8_t *ref_ptr_gray_data, const uint8_t *tar_ptr_gray_data, const uint32_t width, int u, int v, int half_ws) {

    float cost = 0.0f;

    // Patch loop
    for (int dy = -half_ws; dy <= half_ws; dy++) {

        for (int dx = -half_ws; dx <= half_ws; dx++) {
            // cost += std::abs(*(ref_row + u + dx) - *(tar_row + u + dx));
            // cost += std::abs(*(ref_ptr_gray_data + v*width + u + dx) - *(tar_ptr_gray_data + v*width + u + dx));
            cost += std::abs(ref_ptr_gray_data[v*width + u + dx] - tar_ptr_gray_data[v*width + u + dx]);

        }
    }
    return cost;
}

void ptr_with_sad(const uint8_t *ref_ptr_gray_data, const uint8_t *tar_ptr_gray_data, const uint32_t width, const uint32_t height, const uint32_t max_step)
{
    int half_ws = 2;
    float total_cost = 0;
    std::vector<float> costs;
    for (int v = half_ws; v < height-half_ws; v++) {
        for (int u = half_ws; u < width - half_ws; u++) {
            for(uint32_t s=0; s < max_step; s++){

                int current_u = u+s;
                int current_v = v;
                if(current_u >=half_ws && current_v >= half_ws && current_u < width - half_ws - 1 && current_v < height - half_ws - 1){
                    
                    float cost = SADPatch_with_ptr(ref_ptr_gray_data, tar_ptr_gray_data, width, current_u, current_v, half_ws);
                    total_cost += cost;
                    // costs.push_back(cost);
                }
            }
            // std::cout <<"(u,v) " << u << "," << v << "costs:" << costs << std::endl;
        }
    }
    std::cout << "total cost:" << total_cost << " size of costs " << costs.size() << std::endl;

}


int main(int argc, char** argv)
{

    std::string ref_image_path = (argc > 1) ? argv[1] : "../../data/BotanicGarden_rectified_1/left_1666059840150278091.png";
    std::string tar_image_path = (argc > 2) ? argv[2] : "../../data/BotanicGarden_rectified_1/right_1666059840150278091.png";

    cv::Mat cv_ref_gray_data = cv::imread(ref_image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat cv_tar_gray_data = cv::imread(tar_image_path, cv::IMREAD_GRAYSCALE);


    if (cv_ref_gray_data.empty()) {
        std::cerr << "Error: Could not read reference image from " << ref_image_path << std::endl;
        return -1;
    }

    if (cv_tar_gray_data.empty()) {
        std::cerr << "Error: Could not read target image from " << tar_image_path << std::endl;
        return -1;
    }

    uint32_t height = cv_ref_gray_data.rows;
    uint32_t width = cv_ref_gray_data.cols;

    uint8_t *ref_ptr_gray_data = new uint8_t[width*height](); 
    uint8_t *tar_ptr_gray_data = new uint8_t[width*height](); 

    uint32_t max_step = 256;


    memcpy(ref_ptr_gray_data, cv_ref_gray_data.data, width*height);
    memcpy(tar_ptr_gray_data, cv_tar_gray_data.data, width*height);


    std::cout << "cv_at_with_sad ";
    measureTime([&](){cv_at_with_sad(cv_ref_gray_data, cv_tar_gray_data, width, height, max_step);});
    std::cout << "cv_ptr_with_sad ";
    measureTime([&](){cv_ptr_with_sad(cv_ref_gray_data, cv_tar_gray_data, width, height, max_step);});
    std::cout << "ptr_with_sad ";
    measureTime([&](){ptr_with_sad(ref_ptr_gray_data, tar_ptr_gray_data, width, height, max_step);});


}
