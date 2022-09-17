#pragma once

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "def.h"
#include "util.cuh"

namespace clustering {

// -----------------------------------------------------------------------------
int* shift_hash_results(            // shift e2lsh hash results
    int size,                           // number of MPIs (size)
    int n,                              // number of data points
    int m,                              // number of hash tables
    int *hash_results);                 // hash results (return)

// -----------------------------------------------------------------------------
int hash_results_to_buckets(        // convert local hash results into buckets
    int rank,                           // MPI rank
    int size,                           // number of MPIs (size)
    int n,                              // number of data points
    int m,                              // signature size
    int t,                              // threshold of #point IDs in a bucket
    int *hash_results,                  // hash results (allow modify)
    std::vector<int> &bktset,           // bucket set (return)
    std::vector<u64> &bktpos);          // bucket position (return)

// -----------------------------------------------------------------------------
int bins_to_buckets(                // convert global bins into local buckets
    int rank,                           // MPI rank
    int size,                           // number of MPIs (size)
    int n,                              // number of bins
    std::vector<int> &binset,           // bin set (allow modify)
    std::vector<u64> &binpos,           // bin position (allow modify)
    std::vector<int> &bktset,           // bucket set (return)
    std::vector<u64> &bktpos);          // bucket position (return)

} // end namespace clustering
