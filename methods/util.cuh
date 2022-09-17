#pragma once

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

#include <stdint.h>
#include <unistd.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>

#include <omp.h>
#include <mpi.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "def.h"

namespace clustering {

extern int g_k;                     // global param: #clusters
extern f32 g_mae;                   // global param: mean absolute error
extern f32 g_mse;                   // global param: mean square   error
extern f64 g_tot_wc_time;           // global param: total wall clock time (s)
extern f64 g_tot_cpu_time;          // global param: total cpu time (s)

extern int g_num_buckets;           // global param: local number of buckets
extern int g_tot_buckets;           // global param: total number of buckets
extern int g_tot_bins;              // global param: total number of bins
extern int g_tot_seeds;             // global param: total number of seeds
extern f64 g_phase1_wc_time;        // global param: phase 1 wall clock time (s)
extern f64 g_phase2_wc_time;        // global param: phase 2 wall clock time (s)
extern f64 g_phase3_wc_time;        // global param: phase 3 wall clock time (s)
extern f64 g_eval_wc_time;          // global param: eval wall clock time (s)
extern f64 g_eval_cpu_time;         // global param: eval cpu time (s)
extern f64 g_silk_wc_time;          // global param: silk wall clock time (s)
extern f64 g_silk_cpu_time;         // global param: silk cpu time (s)

extern int g_iter;                  // global param: #iterations
extern f64 g_init_wc_time;          // global param: init wall clock time (s)
extern f64 g_init_cpu_time;         // global param: init cpu time (s)
extern f64 g_iter_wc_time;          // global param: iter wall clock time (s)
extern f64 g_iter_cpu_time;         // global param: iter cpu time (s)
extern f64 g_kfreqitems_wc_time;    // global param: k-freqitems wall clock time (s)
extern f64 g_kfreqitems_cpu_time;   // global param: k-freqitems cpu time (s)

// -----------------------------------------------------------------------------
void create_dir(                    // create dir if the path does not exist
    char *path);                        // input path

// -----------------------------------------------------------------------------
void init_mpi_comm(                 // initialize mpi communication
    MPI_INFO &mpi_info);                // mpi_info (return)

// -----------------------------------------------------------------------------
void finalize_mpi_comm(             // finalize mpi communication
    const MPI_INFO &mpi_info);          // mpi_info

// -----------------------------------------------------------------------------
template<class DType>
DType* read_dense_data(             // read dense data (binary) from disk
    int   rank,                         // mpi rank
    int   n,                            // number of data points
    int   d,                            // data dimension
    const char *prefix)                 // prefix of dataset
{
    double start_time = omp_get_wtime();
    std::ios::sync_with_stdio(false);

    char fname[200]; sprintf(fname, "%s_%d.bin", prefix, rank);
    FILE *fp = fopen(fname, "rb");
    if (!fp) { printf("Rank #%d: cannot open %s\n", rank, fname); exit(1); }

    // read dataset from disk
    DType *dataset = new DType[(u64) n*d];
    fread(dataset, sizeof(DType), (u64) n*d, fp);
    fclose(fp);
    
    double loading_time = omp_get_wtime() - start_time;
    printf("Rank #%d: n=%d, d=%d, time=%.2lf seconds, path=%s\n\n", rank, n, 
        d, loading_time, fname);
    return dataset;
}

// -----------------------------------------------------------------------------
template<class DType>
DType* read_sparse_data(            // read sparse data (binary) from disk
    int   rank,                         // mpi rank
    int   n,                            // number of data points
    int   d,                            // data dimension
    const char *prefix,                 // prefix of data set
    u64   *datapos)                     // data position (return)
{
    double start_time = omp_get_wtime();
    std::ios::sync_with_stdio(false);

    char fname[200]; sprintf(fname, "%s_%d.bin", prefix, rank);
    FILE *fp = fopen(fname, "rb");
    if (!fp) { printf("Rank #%d: cannot open %s\n", rank, fname); exit(1); }

    // read the start position of each data
    fread(datapos, sizeof(u64), n+1, fp);
    
    // read dataset
    u64 N = datapos[n];
    DType *dataset = new DType[N];
    fread(dataset, sizeof(DType), N, fp);
    fclose(fp);
    
    double loading_time = omp_get_wtime() - start_time;
    printf("Rank #%d: n=%d, d=%d, N=%lu, time=%.2lf seconds, path=%s\n\n", 
        rank, n, d, N, loading_time, fname);
    return dataset;
}

// -----------------------------------------------------------------------------
template<class DType>
__host__ __device__ float l2_sqr(   // calc the square of euclidean distance
    int   dim,                          // data dimension
    const DType *data,                  // data point
    const float *centroid)              // centroid
{
    float dist = 0.0f;
    for (int j = 0; j < dim; ++j) {
        dist += SQR(centroid[j] - data[j]);
    }
    return dist;
}

// -----------------------------------------------------------------------------
template<class DType>
__host__ __device__ float jaccard_dist(// calc jaccard distance
    int   n_data,                       // number of data dimensions
    int   n_mode,                       // number of mode dimensions
    const DType *data,                  // data point
    const int   *mode)                  // mode
{
    int overlap = 0, i = 0, j = 0; // i for data, j for mode
    while (i < n_data && j < n_mode) {
        if (data[i] < mode[j]) ++i;
        else if (mode[j] < data[i]) ++j;
        else { ++overlap; ++i; ++j; }
    }
    return 1.0f - (float) overlap / (n_data + n_mode - overlap);
}

// -----------------------------------------------------------------------------
__host__ __device__ inline int get_length(// get the length of pos
    int   id,                           // input id
    const u64 *pos)                     // pos array
{
    return int(pos[id+1] - pos[id]);
}

// -----------------------------------------------------------------------------
template<class DType>
int distinct_coord_and_freq(        // get max freq, distinct coords & freqs
    u64   total_num,                    // total number of coordinates
    DType *arr,                         // store all coordinates (allow modify)
    DType *coord,                       // distinct coordinates (return)
    int   *freq,                        // frequency (return)
    int   &cnt)                         // counter for #distinct (return)
{
    // sort all coordinates in ascending order
    thrust::sort(arr, arr + total_num);
    
    // get the distinct coordinates and their frequencies (sequential)
    int max_freq = 0, last = 0, this_freq = -1;
    cnt = 0;
    for (size_t i = 1; i < total_num; ++i) {
        if (arr[i] != arr[i-1]) {
            this_freq = i - last;
            coord[cnt] = arr[i-1]; freq[cnt] = this_freq;
            if (this_freq > max_freq) max_freq = this_freq;
            
            last = i; ++cnt;
        }
    }
    // deal with the last element of arr
    this_freq = total_num - last;
    coord[cnt] = arr[total_num-1]; freq[cnt] = this_freq;
    if (this_freq > max_freq) max_freq = this_freq;
    ++cnt;
    
    return max_freq;
}

// -----------------------------------------------------------------------------
void copy_pos(                      // copy a partial datapos to another pos
    int   n,                            // length of partial datapos
    const u64* datapos,                 // partial datapos
    u64   *pos);                        // another pos (return)

// -----------------------------------------------------------------------------
void all_buff_to_cand_list(         // convert all buffers into cand list
    int   n,                            // length of all buffers
    const int *all_buff,                // all buffers
    std::unordered_map<int, std::vector<std::pair<int, int> > > &cand_list);

// -----------------------------------------------------------------------------
void all_buff_to_cand_list(         // convert all buffers into cand list
    int   n,                            // length of all buffers
    const int *all_buff,                // all buffers
    std::unordered_map<int, std::vector<std::pair<int, int> > > &cand_list,
    std::unordered_map<int, int> &cand_cnt);

// -----------------------------------------------------------------------------
u64 get_total_coords(               // get total number of coordinates
    int size,                           // number of MPIs (size)
    u64 num_coords);                    // number of coordinates (local)

// -----------------------------------------------------------------------------
int get_total_buckets(              // get total number of buckets
    int size,                           // number of MPIs (size)
    int num_buckets);                   // number of buckets (local)

// -----------------------------------------------------------------------------
int get_max_round(                  // broadcast num_buffer to get max round
    int size,                           // number of MPIs (size)
    int num_buffer);                    // number of buffer (local)

// -----------------------------------------------------------------------------
int gather_all_buffers(             // gather buffers from diff threads to root
    int   size,                         // number of MPIs (size)
    const std::vector<int> &buffer,     // buffer in local
    int   *all_buff);                   // all buffers @root (return)

// -----------------------------------------------------------------------------
void broadcast_set_and_pos(         // broadcast set and pos to other threads
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    std::vector<int> &binset,           // bin set (return)
    std::vector<u64> &binpos);          // bin position (return)

// -----------------------------------------------------------------------------
//	Given left margin (uint32_t min) and right margin (uint32_t max), generate 
//  a random 32-bits unsigned integer from uniform(min, max).
// -----------------------------------------------------------------------------
u32 uniform_u32(                    // gen a random variable from uniform u32
    u32 min,                            // min value
    u32 max);                           // max value

// -----------------------------------------------------------------------------
float uniform(                      // gen a random variable from uniform distr.
    float start,                        // start position
    float end);                         // end position

// -----------------------------------------------------------------------------
//  generate a random variable from standard Guassian distribution N(0,1)
//  use Box-Muller method
// -----------------------------------------------------------------------------
float gaussian();                   // gen a random variable from N(0,1)

// -----------------------------------------------------------------------------
float cauchy();                     // gen a random variable from Cauchy(1,0)

// -----------------------------------------------------------------------------
void syn_hash_params(               // synchronize hash parameters
    int   len_s,                        // length of random projection
    int   len_p,                        // length of random shift
    float *shift,                       // random shift (return)
    float *proj);                       // random projection (return)

// -----------------------------------------------------------------------------
void output_buckets(                // output buckets to disk
    int   rank,                         // mpi rank
    int   n,                            // number of buckets
    const int  *bktset,                 // bucket set
    const u64  *bktpos,                 // bucket position
    const char *prefix);                // prefix path

// -----------------------------------------------------------------------------
void output_bins(                   // output bins to disk
    int   rank,                         // mpi rank
    int   n,                            // number of bins
    const int  *binset,                 // bin set
    const u64  *binpos,                 // bin position
    const char *prefix);                // prefix path

// -----------------------------------------------------------------------------
void output_centroids(              // output centroids to disk
    int   rank,                         // mpi rank
    int   len,                          // len = k * d
    const float *centroids,             // centroids
    const char  *prefix);               // prefix
    
// -----------------------------------------------------------------------------
template<class DType>
void output_modes(                   // output bins to disk
    int   rank,                         // mpi rank
    int   k,                            // number of modes
    const DType *modeset,               // mode set
    const u64   *modepos,               // mode position
    const char  *prefix)                // prefix path
{
    if (rank > 0) return;
    
    char fname[100]; sprintf(fname, "%s_modes", prefix);
    FILE *fp = fopen(fname, "wb");
    if (!fp) { printf("Could not open %s\n", fname); exit(1); }

    fwrite(&k,      sizeof(int),   1,          fp);
    fwrite(modepos, sizeof(u64),   k+1,        fp);
    fwrite(modeset, sizeof(DType), modepos[k], fp);
    fclose(fp);
}

} // end namespace clustering
