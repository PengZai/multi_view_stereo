#include "persistent_homology.h"




namespace MVS
{





MinimumPeak::MinimumPeak(int idx)
{
    born_idx_ = idx;
    left_edge_idx_ = idx;
    right_edge_idx_ = idx;
    died_idx_ = -1;
}

float MinimumPeak::get_persistence(const std::vector<float>& costs) const
{
    return std::abs(costs[born_idx_] - costs[died_idx_]);
}



void get_peaks_with_persistent_homology(const std::vector<float>& costs, std::vector<MinimumPeak>& minimum_peaks)
{

    int Ncosts = costs.size();
    std::vector<int> idxtopeak(Ncosts, -1);

    std::vector<size_t> indexes(Ncosts);
    std::iota(indexes.begin(), indexes.end(), 0);


    // Process each sample in ascending order 
    std::sort(indexes.begin(), indexes.end(), [&](size_t i, size_t j) {
              return costs[i] < costs[j];
          });


    for(size_t& idx : indexes)
    {
        // is it left direction has been aligned with a minimum peak?
        bool is_left_idx_aligned_with_peak = false;
        int peak_idx_on_the_idx_left = -1;

        if(idx > 0 && idxtopeak[idx-1] != -1){
            is_left_idx_aligned_with_peak = true;
            peak_idx_on_the_idx_left = idxtopeak[idx-1];

        }

        // is it right direction has been aligned with a minimum peak?
        bool is_right_idx_aligned_with_peak = false;
        int peak_idx_on_the_idx_right = -1;

        if(idx < Ncosts-1 && idxtopeak[idx+1] != -1){
            is_right_idx_aligned_with_peak = true;
            peak_idx_on_the_idx_right = idxtopeak[idx+1];
        }        

        // New peak born
        if(is_left_idx_aligned_with_peak == false && is_right_idx_aligned_with_peak == false){
            minimum_peaks.emplace_back(MinimumPeak(idx));
            idxtopeak[idx] = minimum_peaks.size()-1;
            if(minimum_peaks.size() == 1){
                minimum_peaks[0].died_idx_ = indexes[Ncosts - 1];
            }
        }
  

        // Directly merge to next peak left
        if(is_left_idx_aligned_with_peak == true && is_right_idx_aligned_with_peak == false)
        {
            minimum_peaks[peak_idx_on_the_idx_left].right_edge_idx_ += 1;
            idxtopeak[idx] = peak_idx_on_the_idx_left;
        }

        // Directly merge to next peak right
        if(is_left_idx_aligned_with_peak == false && is_right_idx_aligned_with_peak == true)
        {
            minimum_peaks[peak_idx_on_the_idx_right].left_edge_idx_ -= 1;
            idxtopeak[idx] = peak_idx_on_the_idx_right;
        }

        if(is_left_idx_aligned_with_peak == true && is_right_idx_aligned_with_peak == true)
        {
            // left was born earlier: merge right to left
            if( costs[minimum_peaks[peak_idx_on_the_idx_left].born_idx_] < costs[minimum_peaks[peak_idx_on_the_idx_right].born_idx_])
            {
                minimum_peaks[peak_idx_on_the_idx_right].died_idx_ = idx;
                minimum_peaks[peak_idx_on_the_idx_left].right_edge_idx_ = minimum_peaks[peak_idx_on_the_idx_right].right_edge_idx_;
                idxtopeak[idx] = peak_idx_on_the_idx_left;
                idxtopeak[minimum_peaks[peak_idx_on_the_idx_left].right_edge_idx_] = peak_idx_on_the_idx_left;
            }
            else{

                minimum_peaks[peak_idx_on_the_idx_left].died_idx_ = idx;
                minimum_peaks[peak_idx_on_the_idx_right].left_edge_idx_ = minimum_peaks[peak_idx_on_the_idx_left].left_edge_idx_;
                idxtopeak[idx] = peak_idx_on_the_idx_right;
                idxtopeak[minimum_peaks[peak_idx_on_the_idx_right].left_edge_idx_] = peak_idx_on_the_idx_right;  
            }
        }

    }


    // Process each sample in descending order 
    std::sort(minimum_peaks.begin(), minimum_peaks.end(), [&](const MinimumPeak& minimum_peak_i, const MinimumPeak& minimum_peak_j) {
              return minimum_peak_i.get_persistence(costs) > minimum_peak_j.get_persistence(costs);
    });

}


}