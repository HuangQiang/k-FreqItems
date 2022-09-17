#pragma once

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <vector>
#include <unordered_map>

#include "def.h"
#include "util.cuh"

namespace clustering {

// -----------------------------------------------------------------------------
int signatures_to_bins(             // convert local signatures into global bins
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   N,                            // total number of data points
    int   n,                            // num of buckets / length of signatures
    int   l,                            // number of minhash tables
    int   b,                            // threshold of #bucket IDs in a bin
    int   delta,                        // threshold of #point  IDs in a bin
    bool  filter,                       // whether filter #point IDs in a bin
    float gbeta,                        // global beta
    float lbeta,                        // local  beta
    const int *bktset,                  // bucket set
    const u64 *bktpos,                  // bucket position
    int   *signatures,                  // minhash signatures (allow modify)
    std::vector<int> &binset,           // bin set (return)
    std::vector<u64> &binpos);          // bin position (return)

// -----------------------------------------------------------------------------
u64 labels_to_index(                // convert labels into index and index_pos
    int   size,                         // number of MPIs (size)
    int   n,                            // number of labels
    int   k,                            // number of centers
    const int *labels,                  // cluster labels for data points
    std::vector<int> &binset,           // bin set (return)
    std::vector<u64> &binpos);          // bin position (return)

// -----------------------------------------------------------------------------
int labels_to_bins(                 // labels to bins & re-number labels
    int rank,                           // MPI rank
    int size,                           // number of MPIs (size)
    int n,                              // number of data points
    int k,                              // number of centers
    int *labels,                        // cluster labels for data (return)
    std::vector<int> &binset,           // bin set (return)
    std::vector<u64> &binpos);          // bin position (return)

} // end namespace clustering
