#include "undistort.h"



namespace MVS
{

DistortModel::DistortModel(int width, int height, int wOrg, int hOrg):
width_(width),
height_(height),
wh_(width*height),
wOrg_(wOrg),
hOrg_(hOrg)
{


  remapU_ = new float[wh_]();
  remapV_ = new float[wh_]();

}

DistortModel::~DistortModel()
{
    delete[] remapU_;
    delete[] remapV_;
}

void DistortModel::resetRemapUV()
{
    for(int v=0;v<height_;v++)
    {
        for(int u=0;u<width_;u++)
        {
            int coords = v*width_+u;
            remapU_[coords] = u;
            remapV_[coords] = v;
        }
    }
}


void DistortModel::makeResizeK(std::vector<float> original_intrinsics, std::vector<float>& intrinsics)
{



    float xscale = (float)width_/wOrg_;
    float yscale = (float)height_/hOrg_;

    intrinsics[0] = original_intrinsics[0]*xscale;
    intrinsics[1] = original_intrinsics[1]*yscale;
    intrinsics[2] = original_intrinsics[2]*xscale;
    intrinsics[3] = original_intrinsics[3]*yscale;


    resetRemapUV();

	printf("orginal H: %d, original W:%d, H/W:%.4f!\n", hOrg_, wOrg_, float(hOrg_)/wOrg_);
	printf("H: %d, W:%d, H/W:%.4f!\n", height_, width_, float(height_)/width_);

    printf("new_fx: %.4f, new_fy: %.4f, new_cx: %.4f, new_cy: %.4f\n", intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]);

    distortCoordinates(remapU_, remapV_, wh_, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3], remapU_, remapV_);


}


void DistortModel::makeOptimalKCrop(std::vector<float>& intrinsics)
{


    resetRemapUV();
  

    float* tgX = new float[100000]();
	float* tgY = new float[100000]();
	float minX = 0;
	float maxX = 0;
	float minY = 0;
	float maxY = 0;

    // convert into normal space between [-5,5]
	for(int x=0; x<100000;x++)
    {
        tgX[x] = (x-50000.0f) / 10000.0f; 
        tgY[x] = 0;
    }
    distortCoordinates(tgX, tgY, 100000, unit_fx_, unit_fy_, unit_cx_, unit_cy_, tgX, tgY);
    for(int x=0; x<100000;x++)
        {
            if(tgX[x] > 0 && tgX[x] < wOrg_-1)
            {
                if(minX==0) minX = (x-50000.0f) / 10000.0f;
                maxX = (x-50000.0f) / 10000.0f;
            }
        }
    for(int y=0; y<100000;y++)
        {
        tgY[y] = (y-50000.0f) / 10000.0f;
        tgX[y] = 0;
    }
    distortCoordinates(tgX, tgY, 100000, unit_fx_, unit_fy_, unit_cx_, unit_cy_, tgX, tgY);
    for(int y=0; y<100000;y++)
    {
        if(tgY[y] > 0 && tgY[y] < hOrg_-1)
        {
            if(minY==0) minY = (y-50000.0f) / 10000.0f;
            maxY = (y-50000.0f) / 10000.0f;
        }
    }

    delete[] tgX;
	delete[] tgY;

	minX *= 1.01;
	maxX *= 1.01;
	minY *= 1.01;
	maxY *= 1.01;

	printf("initial range: x: %.4f - %.4f; y: %.4f - %.4f!\n", minX, maxX, minY, maxY);


    bool hasBlackLeft = true, hasBlackRight = true, hasBlackTop = true, hasBlackBottom = true;
    int iteration_count = 0;
    while(hasBlackLeft || hasBlackRight || hasBlackBottom || hasBlackTop)
    {
        hasBlackLeft = hasBlackRight = hasBlackTop = hasBlackBottom = false;
        for(int v=0;v<height_;v++)
        {
        remapU_[v] = minX;
        remapU_[v+height_] = maxX;
        remapV_[v] = remapV_[v+height_] = minY + (maxY-minY)*(float)v/(height_-1.0f);
        }
        distortCoordinates(remapU_, remapV_, 2*height_, unit_fx_, unit_fy_, unit_cx_, unit_cy_, remapU_, remapV_);

        for(int v=0;v<height_;v++)
        {
    
            if(remapU_[v] <= 0 || remapU_[v] >= wOrg_-1){
            hasBlackLeft = true;
            }
            if(remapU_[v+height_] <= 0 || remapU_[v+height_] >= wOrg_-1){
            hasBlackRight = true;
            }

        }


        for(int u=0;u<width_;u++)
        {
        remapV_[u] = minY;
        remapV_[u+width_] = maxY;
        remapU_[u] = remapU_[u+width_] = minX + (maxX-minX)*(float)u/(width_-1.0f);
        }
        distortCoordinates(remapU_, remapV_, 2*width_, unit_fx_, unit_fy_, unit_cx_, unit_cy_, remapU_, remapV_);

        for(int u=0;u<width_;u++)
        {
    
            if(remapV_[u] <= 0 || remapV_[u] >= hOrg_-1){
            hasBlackTop = true;
            }
            if(remapV_[u+width_] <= 0 || remapV_[u+width_] >= hOrg_-1){
            hasBlackBottom = true;
            }

        }

        if((hasBlackLeft || hasBlackRight) && (hasBlackTop || hasBlackBottom))
        {
        if((maxX-minX) > (maxY-minY))
            hasBlackBottom = hasBlackTop = false;	// only shrink left/right
        else
            hasBlackLeft = hasBlackRight = false; // only shrink top/bottom
        }

        if(hasBlackLeft) minX *= 0.995;
        if(hasBlackRight) maxX *= 0.995;
        if(hasBlackTop) minY *= 0.995;
        if(hasBlackBottom) maxY *= 0.995;

        iteration_count++;


        printf("iteration %05d: range: x: %.4f - %.4f; y: %.4f - %.4f!\n", iteration_count,  minX, maxX, minY, maxY);
        if(iteration_count > 500)
        {
            printf("FAILED TO COMPUTE GOOD CAMERA MATRIX - SOMETHING IS SERIOUSLY WRONG. ABORTING \n");
            std::exit(1);
        }


    } 

    float h_by_w  = float(hOrg_)/wOrg_;
    float lengthY = maxY - minY;
    float middleY = minY + lengthY/2;
    float lengthX = maxX - minX;
    float middleX = minX + lengthX/2;

	printf("orginal H: %d, original W:%d, H/W:%.4f!\n", hOrg_, wOrg_, float(hOrg_)/wOrg_);

	printf("initial range: x: %.4f - %.4f; y: %.4f - %.4f, h: %.4f, w: %.4f, h/w:%.4f!\n", minX, maxX, minY, maxY, maxY-minY, maxX-minX, (maxY-minY)/(maxX-minX));

    // recalculate intrinsics
    float fx;
    float fy;
    float cx;
    float cy;

    // keep original h_by_w  
    if(lengthX > lengthY){

        float correct_lengthX = lengthY / h_by_w;
        minX = middleX - correct_lengthX/2;
        maxX = middleX + correct_lengthX/2;
        fx = ((float)width_)/(correct_lengthX);
        fy = ((float)height_)/(lengthY);
        cx = -minX*fx;
        cy = -minY*fy;

    }
    else{
        float correct_lengthY = lengthX * h_by_w;
        minY = middleY - correct_lengthY/2;
        maxY = middleY + correct_lengthY/2;
        fx = ((float)width_)/(lengthX);
        fy = ((float)height_)/(correct_lengthY);
        cx = -minX*fx;
        cy = -minY*fy;
    }

    intrinsics[0] = fx;
    intrinsics[1] = fy;
    intrinsics[2] = cx;
    intrinsics[3] = cy;

    // keep pixel as much as possible
	// new_fx_ = ((float)new_w_)/(maxX-minX);
	// new_fy_ = ((float)new_h_)/(maxY-minY);
	// new_cx_ = -minX*new_fx_;
	// new_cy_ = -minY*new_fy_;

	printf("finnal range: x: %.4f - %.4f; y: %.4f - %.4f, h: %.4f, w: %.4f, h/w:%.4f!\n", minX, maxX, minY, maxY, maxY-minY, maxX-minX, (maxY-minY)/(maxX-minX));

    printf("new_fx: %.4f, new_fy: %.4f, new_cx: %.4f, new_cy: %.4f\n", fx, fy, cx, cy);

    resetRemapUV();

    distortCoordinates(remapU_, remapV_, wh_, fx, fy, cx, cy, remapU_, remapV_);

   


}




void DistortModel::undistort(Image* const image)
{

    int wOrg = image->getWidthOrg();
    int hOrg = image->getHeightOrg();
    int width = image->getWidth();
    int height = image->getHeight();

    uint8_t* ptr_raw_gray_data = image->getRawGrayDataPtr();

    PixelPoint* ptr_pixel_point_matrix = image->getPixelPointMatrixPtr();

    for (int v = 0; v < height; v++) 
    {
        for (int u = 0; u < width; u++) 
        {
            int cidx = v*width+u;
            float u_distorted = remapU_[cidx];
            float v_distorted = remapV_[cidx];
            if(u_distorted != -1.0 && v_distorted != -1.0)
            {
                float cidx_distotred = v_distorted*width + u_distorted;
                ptr_pixel_point_matrix[cidx].setUV(u,v);
                ptr_pixel_point_matrix[cidx].setImagePtr(image);

                ptr_pixel_point_matrix[cidx].epipolar_segment_vec_.emplace_back(EpipolarSegment(image->config_->infinite_inv_depth_, 1/image->config_->min_depth_));

                if(cidx_distotred != float(cidx)){

                    ptr_pixel_point_matrix[cidx].setIntensity(getBilinearInterpolated(ptr_raw_gray_data, wOrg, hOrg, u_distorted, v_distorted));

                }
                else{
                    ptr_pixel_point_matrix[cidx].setIntensity(ptr_raw_gray_data[cidx]); 
                }
            }
            else{
                ptr_pixel_point_matrix[cidx].setIntensity(0);
                // std::cout << "u:" << u << ",v:" << v << "is black" << std::endl;
            }

        }
    }

}










// EquidistantDistortModel::EquidistantDistortModel(int new_w, int new_h, int wOrg, int hOrg)
// :DistortModel(new_w, new_h, wOrg, hOrg){}



// void EquidistantDistortModel::distortCoordinates(float* in_u, float* in_v, int n, float in_fx, float in_fy, float in_cx, float in_cy,
//  float* out_u, float* out_v)
// {


//     for (int i = 0; i < n; i++) {
//           // 按照公式，计算点(u,v)对应到畸变图像中的坐标(u_distorted, v_distorted)
//           float x = (in_u[i] - in_cx) / in_fx;
//           float y = (in_v[i] - in_cy) / in_fy;

//           float r_u = std::sqrt(x * x + y * y);


//           // Compute theta (angle from optical axis)
//           float theta = std::atan(r_u);
//           float theta2 = theta * theta;
//           float theta4 = theta2 * theta2;
//           float theta6 = theta4 * theta2;
//           float theta8 = theta4 * theta4;
//           // Distorted angle
//           float thetad = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);

//           // Mapping back to distorted normalized coordinates
//           // double scale = (theta_d / r_u);
//           float scale = (r_u > 1e-8) ? thetad / r_u : 1.0;

//           float x_d = x * scale;
//           float y_d = y * scale;

//           // Convert to pixel coordinates
//           float u_distorted = fx * x_d + cx;
//           float v_distorted = fy * y_d + cy;

//           out_u[i] = u_distorted;
//           out_v[i] = v_distorted;
        
//     }
// }


RadtanDistortModel::RadtanDistortModel(int width, int height, int wOrg, int hOrg, std::vector<float> original_intrinsics, std::vector<float> distortion_coeffs)
:DistortModel(width, height, wOrg, hOrg){


    k1_org_ = distortion_coeffs[0];
    k2_org_ = distortion_coeffs[1];
    p1_org_ = distortion_coeffs[2];
    p2_org_ = distortion_coeffs[3];

    // original intrinsics
    fx_org_ = original_intrinsics[0];
    fy_org_ = original_intrinsics[1];
    cx_org_ = original_intrinsics[2];
    cy_org_ = original_intrinsics[3];

}


void RadtanDistortModel::distortCoordinates(float* in_u, float* in_v, int n, float in_fx, float in_fy, float in_cx, float in_cy,
 float* out_u, float* out_v)
{


    for (int i = 0; i < n; i++) {
          // 按照公式，计算点(u,v)对应到畸变图像中的坐标(u_distorted, v_distorted)

          float x = (in_u[i] - in_cx) / in_fx;
          float y = (in_v[i] - in_cy) / in_fy;
          float r = sqrt(x * x + y * y);
          float x_distorted = x * (1 + k1_org_ * r * r + k2_org_ * r * r * r * r) + 2 * p1_org_ * x * y + p2_org_ * (r * r + 2 * x * x);
          float y_distorted = y * (1 + k1_org_ * r * r + k2_org_ * r * r * r * r) + p1_org_ * (r * r + 2 * y * y) + 2 * p2_org_ * x * y;
          float u_distorted = fx_org_ * x_distorted + cx_org_;
          float v_distorted = fy_org_ * y_distorted + cy_org_;

          out_u[i] = u_distorted;
          out_v[i] = v_distorted;
        
    }
}

} // namespace MVS

