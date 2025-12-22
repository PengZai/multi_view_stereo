#pragma once

#include<iostream>
#include<vector>
#include<numeric>
#include<algorithm>

namespace MVS
{

class MinimumPeak
{
    public:
    MinimumPeak(int idx);
    ~MinimumPeak() = default;

    int born_idx_;
    int died_idx_;
    int left_edge_idx_;
    int right_edge_idx_;

    float get_persistence(const std::vector<float>& costs) const;


};


void get_peaks_with_persistent_homology(const std::vector<float>& costs, std::vector<MinimumPeak>& minimum_peaks);




}
