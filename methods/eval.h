#pragma once

#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>

#include "def.h"
#include "util.cuh"

namespace clustering {

// -----------------------------------------------------------------------------
template<class DType>
float calc_jaccard_dist(            // calc jaccard dist between data & seed
    int   did,                          // data id
    int   sid,                          // label (seed id)
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    const int   *seedset,               // seed set
    const u64   *seedpos)               // seed position
{
    int n_data = get_length(did, datapos);
    const DType *data = dataset + datapos[did];
    
    int n_seed = get_length(sid, seedpos);
    const int *seed = seedset + seedpos[sid];
    
    return jaccard_dist<DType>(n_data, n_seed, data, seed);
}

// -----------------------------------------------------------------------------
template<class DType>
void calc_local_stat_by_seeds(      // calc local statistics by seeds
    int   n,                            // number of data points
    int   k,                            // number of clusters
    const int   *labels,                // cluster labels for data points
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    const int   *seedset,               // seed set
    const u64   *seedpos,               // seed position
    float *c_mae,                       // mean absolute error (return)
    float *c_mse)                       // mean square   error (return)
{
    // calc the jaccard distance for local data to its nearest seed
    float *dist = new float[n];
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        dist[i] = calc_jaccard_dist<DType>(i, labels[i], dataset, datapos, 
            seedset, seedpos);
    }
    
    // update the statistics information
    int   sid = -1;
    float dis = -1.0f;
    for (int i = 0; i < n; ++i) {
        sid = labels[i]; dis = dist[i];
        c_mae[sid] += dis;
        c_mse[sid] += dis*dis;
    }
    delete[] dist;
}

// -----------------------------------------------------------------------------
void calc_global_stat(              // calc global statistics
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   k,                            // number of clusters
    float *c_mae,                       // mean absolute error (allow modify)
    float *c_mse,                       // mean square   error (allow modify)
    float &mae,                         // mean absolute error (return)
    float &mse);                        // mean square error (return)

} // end namespace clustering
