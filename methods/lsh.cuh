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
//  Broder, Andrei Z., Moses Charikar, Alan M. Frieze, and Michael Mitzenmacher.
//  "Min-wise independent permutations." 
//  In Proceedings of the thirtieth annual ACM Symposium on Theory of Computing
//  (STOC), pp. 327-336. 1998.
// -----------------------------------------------------------------------------
template<class DType>
void minhash(                       // minwise hashing (pair-wise)
    int   rank,                         // MPI rank
    int   n,                            // number of data points/buckets
    int   n_prime,                      // prime number for data/buckets
    int   d_prime,                      // prime number for dimension
    int   m,                            // #hash tables
    int   h,                            // #concatenated hash func
    const DType *dataset,               // data/bucket set
    const u64   *datapos,               // data/bucket position
    int   *hash_results);               // hash results (return)

// -----------------------------------------------------------------------------
//  Datar, Mayur, Nicole Immorlica, Piotr Indyk, and Vahab S. Mirrokni. 
//  "Locality-sensitive hashing scheme based on p-stable distributions." 
//  In Proceedings of the twentieth annual Symposium on Computational Geometry
//  (SoCG), pp. 253-262. 2004.
// -----------------------------------------------------------------------------
template<class DType>
void e2lsh(                         // calc hash results using e2lsh
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   d,                            // data dimension
    int   prime,                        // prime number
    int   m,                            // number of hash tables
    int   h,                            // number of concat hash functions
    float w,                            // bucket width
    const float *proj,                  // random projection, m*h*d
    const float *shift,                 // random shift, m*h
    const DType *dataset,               // data set
    int   *hash_results);               // hash results (return)

} // end namespace clustering
