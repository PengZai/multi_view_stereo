#include<iostream>
#include<vector>
#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>
#include <implot.h>



#include "../src/persistent_homology.h"

 

int main(int argc, char** argv) {


    std::vector<float> costs = {
        // 49.2721,37.2313,25.4558,24.0068,22.1066,16.8367,9.55102,6.97959,4.28571,8.69388,8.69388,13.0204,13.0612,14.5306,18.9637,18.4331,20.1882,18.0658,17.8776,15.0204,15.3946,16.7007,16.3129,18.966,19.2925,17.1701,19.1701,18.8844,20.068,20.8027,18.4354,20.1905,19.2925,19.068,20.7823,22.4966,23.517,27.966,37.1088,50.7007,64.7007,75.2109,84.9252,136.027,147.905,153.435,158.211,162.578,169.68,174.048,177.313,198.497
25.7188,26.3515,24.1882,23.5714,22.1224,21.898,19.7959,17.0816,17.102,15.9184,17.3673,16.8776,17.6122,18.5102,17.8776,15.3265,17.0204,17.3469,19.1224,16.6327,16.898,17.449,16.7143,18.3469,17.1224,18.6939,22.1429,19.6531,22.5102,21.6939,25.551,25.1429,25.0612,24.3265,23.9592,25.449,19.8163,18.4082,19.4286,20.2857,19.2857,20.4082,19.1224,18.8163,16.9388,17.7551,15.3878,17.5918,17.3878,16.5714,18.7551,16.5102,17.5306,18.8367,18,16.4898,19.4286,18.3673,19.0612,18.7347,19.3061,19.3878,19.6531,18.6735,21.3265,24.1633,24.0408,24.0408,25.9592,22.898,24.7959,23.6327,27.2857,27.9796,26.8776,25,25.7347,27.9796,35.3469,31.1837,35.7143,33.3061,27.8776,28.7347,29.1837,26.898,28.6531,29.4898,29.2449,29.551,29.7551,29.8163,33.4082,35.2041,31.0408,36.1837,37.9796,36.3673,37.2245,34.9388,34.449,29.9184,26.8571,28.3265,26.6939,26.8571,26.7551,25.8163,33.8571,33.4286,31.1837,29.6122,22.6327,21.5306,27.4082,35.4082,39.4898,41.1633,43.2857,41.3673,41.6122,26.9592,27,30.3061,31.4082
    };

    std::vector<int> indexes(costs.size());
    std::iota(indexes.begin(), indexes.end(), 0);

    auto [minIt, maxIt] = std::minmax_element(costs.begin(), costs.end());

    float min_cost = *minIt;
    float max_cost = *maxIt;

    float acceptable_threshold = (1+0.1) * min_cost;
    float margin_pers = 0.2 * min_cost;

    std::vector<MVS::MinimumPeak> minimum_peaks;
    

    MVS::get_peaks_with_persistent_homology(costs, minimum_peaks);

    auto viewer = guik::viewer(Eigen::Vector2i(-1, -1), false, "viewer");

    viewer->register_ui_callback("ui_callback", [&]() {

        ImGui::Begin("Cost Curve");
        if (ImPlot::BeginPlot("Cost Plots")) {

            ImPlot::SetupAxes("step","cost");
            std::string cost_label = "cost ";
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
            ImPlot::PlotLine(cost_label.data(), costs.data(), costs.size());

            for(int i=0; i<(int)minimum_peaks.size();i++)
            {
                MVS::MinimumPeak& minimum_peak = minimum_peaks[i];
                float born_cost = costs[minimum_peak.born_idx_];
                float pers = minimum_peak.get_persistence(costs);
                if(born_cost <= acceptable_threshold && pers >= margin_pers){
                    std::string minimum_peak_label = "minimum_peak" + std::to_string(i);
                    ImPlot::PlotInfLines(minimum_peak_label.data(), indexes.data() + minimum_peak.born_idx_, 1);
                }
            }

            ImPlot::EndPlot();
        }

        ImGui::End();


    });


    while (viewer->spin_once()) {

    }


    return 0;


}