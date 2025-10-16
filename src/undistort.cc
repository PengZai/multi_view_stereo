#include "undistort.h"



namespace MVS
{

Undistort::Undistort()
{

}

Undistort::~Undistort()
{
    delete[] remapU_;
    delete[] remapV_;
}

void Undistort::run(const Image* image)
{
    int width = image->getWidth();
    int height = image->getHeight();

    uint8_t* ptr_raw_gray_data = image->getRawGrayDataPtr();

    float* ptr_gray_data = image->getGrayDataPtr();

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
                if(cidx_distotred != float(cidx)){
                    ptr_gray_data[cidx] = getBilinearInterpolated(ptr_raw_gray_data, width, height, u_distorted, v_distorted);
                }
                else{
                    ptr_gray_data[cidx] = ptr_raw_gray_data[cidx];
                }
            }
            else{
                ptr_gray_data[cidx] = 0;
            }

        }
    }

}

void Undistort::createRemapWithUndistortedModel(int width, int height)
{
    remapU_ = new float[width*height]();
    remapV_ = new float[width*height]();

    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
                remapU_[v*width+u] = u;
                remapV_[v*width+u] = v;
        }
    }
}

void Undistort::createRemapWithPinholeAndRadtan(int width, int height, const std::vector<float>& intrinsics, const std::vector<float>& distortion_coeffs)
{


    remapU_ = new float[width*height]();
    remapV_ = new float[width*height]();

    float fx = intrinsics[0];
    float fy = intrinsics[1];
    float cx = intrinsics[2];
    float cy = intrinsics[3];

    float k1 = distortion_coeffs[0];
    float k2 = distortion_coeffs[1];
    float p1 = distortion_coeffs[2];
    float p2 = distortion_coeffs[3];

    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
            double x = (u - cx) / fx, y = (v - cy) / fy;
            double r = sqrt(x * x + y * y);
            double x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
            double y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
            double u_distorted = fx * x_distorted + cx;
            double v_distorted = fy * y_distorted + cy;

            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < width && v_distorted < height) {
                remapU_[v*width+u] = u_distorted;
                remapV_[v*width+u] = v_distorted;

            } else {
                remapU_[v*width+u] = -1.0;
                remapV_[v*width+u] = -1.0;

            }
        }
  }


}

void Undistort::createRemapWithPinholeAndFOV(int width, int height, const std::vector<float>& intrinsics, const std::vector<float>& distortion_coeffs)
{


    remapU_ = new float[width*height]();
    remapV_ = new float[width*height]();

    float fx = intrinsics[0];
    float fy = intrinsics[1];
    float cx = intrinsics[2];
    float cy = intrinsics[3];

    float omega = distortion_coeffs[0];
	float tan_omega = 2.0f * tan(omega / 2.0f);

    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
            double x = (u - cx) / fx, y = (v - cy) / fy;
            double r = sqrt(x * x + y * y);
            float fac = (r==0 || omega==0) ? 1 : atanf(r * tan_omega)/(omega*r);

            double x_distorted = fac * x;
            double y_distorted = fac * y;
            double u_distorted = fx * x_distorted + cx;
            double v_distorted = fy * y_distorted + cy;

            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < width && v_distorted < height) {
                remapU_[v*width+u] = u_distorted;
                remapV_[v*width+u] = v_distorted;

            } else {
                remapU_[v*width+u] = -1.0;
                remapV_[v*width+u] = -1.0;

            }
        }
  }


}
    
} // namespace MVS

