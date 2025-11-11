#include "utils.h"



namespace MVS
{


// --- Basic .npy parser ---
// NpyArray loadNpy(const std::string& path) {
//     std::ifstream f(path, std::ios::binary);
//     if (!f.is_open())
//         throw std::runtime_error("Cannot open file: " + path);

//     // Read magic string
//     char magic[6];
//     f.read(magic, 6);
//     if (std::string(magic, 6) != "\x93NUMPY")
//         throw std::runtime_error("Not a valid .npy file");

//     // Read version
//     unsigned char major, minor;
//     f.read(reinterpret_cast<char*>(&major), 1);
//     f.read(reinterpret_cast<char*>(&minor), 1);

//     // Read header length
//     uint16_t header_len16 = 0;
//     uint32_t header_len32 = 0;
//     size_t header_len = 0;
//     if (major == 1) {
//         f.read(reinterpret_cast<char*>(&header_len16), 2);
//         header_len = header_len16;
//     } else {
//         f.read(reinterpret_cast<char*>(&header_len32), 4);
//         header_len = header_len32;
//     }

//     // Read header
//     std::string header(header_len, ' ');
//     f.read(&header[0], header_len);

//     // Extract shape
//     std::regex shape_re("\\(([^\\)]*)\\)");
//     std::smatch match;
//     std::vector<size_t> shape;
//     if (std::regex_search(header, match, shape_re)) {
//         std::stringstream ss(match[1]);
//         std::string item;
//         while (std::getline(ss, item, ',')) {
//             size_t dim = std::stoul(item);
//             if (dim > 0) shape.push_back(dim);
//         }
//     }

//     // Extract dtype
//     std::regex dtype_re("'descr': *'([^']+)'");
//     std::smatch dtype_match;
//     std::string dtype;
//     if (std::regex_search(header, dtype_match, dtype_re))
//         dtype = dtype_match[1];
//     else
//         throw std::runtime_error("Cannot find dtype in header");

//     // Remove endian indicator (<f4, <u2, etc.)
//     if (dtype.size() > 2 && (dtype[0] == '<' || dtype[0] == '>'))
//         dtype = dtype.substr(1);

//     // Compute number of elements
//     size_t numel = 1;
//     for (auto s : shape) numel *= s;

//     size_t type_size = 0;
//     if (dtype == "float32" || dtype == "f4")
//         type_size = 4;
//     else if (dtype == "uint16" || dtype == "u2")
//         type_size = 2;
//     else
//         throw std::runtime_error("Unsupported dtype: " + dtype);

//     std::vector<unsigned char> buffer(numel * type_size);
//     f.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

//     NpyArray arr;
//     arr.shape = shape;
//     arr.raw_data = std::move(buffer);
//     arr.dtype = dtype;
//     return arr;
// }

float getBilinearInterpolated(const cv::Mat& img, float u, float v) 
{
    int x = floor(u);
    int y = floor(v);

    if (x < 0 || x >= img.cols-1 || y < 0 || y >= img.rows-1)
        return 0.0f; // outside image, return 0 (or handle differently)

    float dx = u - x;
    float dy = v - y;

    float I00 = img.at<uchar>(y, x);
    float I10 = img.at<uchar>(y, x+1);
    float I01 = img.at<uchar>(y+1, x);
    float I11 = img.at<uchar>(y+1, x+1);

    return (1-dx)*(1-dy)*I00 +
           dx*(1-dy)*I10 +
           (1-dx)*dy*I01 +
           dx*dy*I11;
}


float getBilinearInterpolated(
    const float* gray, int width, int height,
    float u, float v)
{
    // Integer floor coordinates
    int x = static_cast<int>(std::floor(u));
    int y = static_cast<int>(std::floor(v));

    // Check bounds (leave a 1-pixel margin for interpolation)
    if (x < 0 || x >= width - 1 || y < 0 || y >= height - 1)
        return 0.0f;

    // Fractional parts
    float dx = u - x;
    float dy = v - y;

    // Four neighboring pixels
    const float* row0 = gray + y * width;
    const float* row1 = gray + (y + 1) * width;

    float I00 = static_cast<float>(row0[x]);
    float I01 = static_cast<float>(row0[x + 1]);
    float I10 = static_cast<float>(row1[x]);
    float I11 = static_cast<float>(row1[x + 1]);

    // Bilinear interpolation
    float I0 = I00 * (1 - dx) + I01 * dx;
    float I1 = I10 * (1 - dx) + I11 * dx;
    return I0 * (1 - dy) + I1 * dy;
}


float getBilinearInterpolated(
    const uint8_t* gray, int width, int height,
    float u, float v)
{
    // Integer floor coordinates
    int x = static_cast<int>(std::floor(u));
    int y = static_cast<int>(std::floor(v));

    // Check bounds (leave a 1-pixel margin for interpolation)
    if (x < 0 || x >= width - 1 || y < 0 || y >= height - 1)
        return 0.0f;

    // Fractional parts
    float dx = u - x;
    float dy = v - y;

    // Four neighboring pixels
    const uint8_t* row0 = gray + y * width;
    const uint8_t* row1 = gray + (y + 1) * width;

    float I00 = static_cast<float>(row0[x]);
    float I01 = static_cast<float>(row0[x + 1]);
    float I10 = static_cast<float>(row1[x]);
    float I11 = static_cast<float>(row1[x + 1]);

    // Bilinear interpolation
    float I0 = I00 * (1 - dx) + I01 * dx;
    float I1 = I10 * (1 - dx) + I11 * dx;
    return I0 * (1 - dy) + I1 * dy;
}

void getBilinearInterpolatedGradient(
    const float* grad, int width, int height,
    float u, float v,
    float& gx, float& gy)
{
    int x = static_cast<int>(std::floor(u));
    int y = static_cast<int>(std::floor(v));

    if (x < 0 || x >= width - 1 || y < 0 || y >= height - 1)
    {
        gx = gy = 0.0f;
        return;
    }

    float dx = u - x;
    float dy = v - y;

    const float* g00 = &grad[2 * (y * width + x)];
    const float* g01 = &grad[2 * (y * width + (x + 1))];
    const float* g10 = &grad[2 * ((y + 1) * width + x)];
    const float* g11 = &grad[2 * ((y + 1) * width + (x + 1))];

    // interpolate Ix
    float gx0 = g00[0] * (1 - dx) + g01[0] * dx;
    float gx1 = g10[0] * (1 - dx) + g11[0] * dx;
    gx = gx0 * (1 - dy) + gx1 * dy;

    // interpolate Iy
    float gy0 = g00[1] * (1 - dx) + g01[1] * dx;
    float gy1 = g10[1] * (1 - dx) + g11[1] * dx;
    gy = gy0 * (1 - dy) + gy1 * dy;
}


cv::Mat getSubpixelPatch(const cv::Mat img, float u, float v,
                         int width, int height) 
{
    int half_w = width  / 2;
    int half_h = height / 2;

    cv::Mat patch(height, width, CV_32F);

    for (int dy = -half_h; dy <= half_h; dy++) {
        for (int dx = -half_w; dx <= half_w; dx++) {
            float uu = u + dx;
            float vv = v + dy;
            patch.at<float>(dy + half_h, dx + half_w) =
                getBilinearInterpolated(img, uu, vv);
        }
    }
    return patch;
}


void getRoundPixelPatch(const cv::Mat& img, float u, float v,
                         int width, int height, cv::Mat& out_patch) 
{
    int half_w = width  / 2;
    int half_h = height / 2;
    int round_u = round(u);
    int round_v = round(v);

    out_patch = img(cv::Rect(round_u-half_w, round_v-half_h, width, height));

}



Eigen::Matrix4f invertTransform(const Eigen::Matrix4f& T)
{   
    Eigen::Matrix3f R = T.block<3,3>(0,0);
    Eigen::Vector3f t = T.block<3,1>(0,3);

    Eigen::Matrix4f T_inv = Eigen::Matrix4f::Identity();
    T_inv.block<3,3>(0,0) = R.transpose();
    T_inv.block<3,1>(0,3) = -R.transpose() * t;
    return T_inv;
}

}

