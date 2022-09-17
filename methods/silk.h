#pragma once

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <string>

#include "def.h"
#include "util.cuh"
#include "lsh.cuh"
#include "bucket.h"
#include "bin.h"
#include "assign.cuh"
#include "eval.h"

namespace clustering {

// -----------------------------------------------------------------------------
//  SILK: a randomized seeding methods based on similar buckets
// -----------------------------------------------------------------------------
template<class DType>
class SILK {
public:
    SILK(                           // constructor
        int   n,                        // number of data points
        int   d,                        // data dimension
        int   rank,                     // MPI rank
        int   size,                     // number of MPIs (size)
        int   b,                        // threshold of #bucket IDs in a bin
        float gbeta,                    // global beta
        float lbeta,                    // local  beta
        float galpha,                   // global alpha
        float lalpha,                   // local  alpha
        const char  *folder,            // output folder
        const DType *dataset,           // data set
        const u64   *datapos);          // data position
    
    // -------------------------------------------------------------------------
    ~SILK();                        // destructor
    
    // -------------------------------------------------------------------------
    void display();                 // display parameters
    
    // -------------------------------------------------------------------------
    int clustering(                 // generic distributed clustering
        int   m,                        // #hash tables (1st level minhash)
        int   h,                        // #concat hash func (1st level minhash)
        int   t,                        // threshold of #point IDs in a bucket
        int   l,                        // #hash tables (2nd level minhash)
        int   k,                        // #concat hash func (2nd level minhash)
        int   delta);                   // threshold of #point IDs in a bin
    
protected:
    int   n_;                       // number of data points
    int   d_;                       // data dimension
    int   rank_;                    // MPI rank
    int   size_;                    // number of MPIs (size)
    int   b_;                       // threshold of #bucket IDs in a bin
    float gbeta_;                   // global beta
    float lbeta_;                   // local  beta
    float galpha_;                  // global alpha
    float lalpha_;                  // local  alpha
    char  folder_[200];             // output folder
    const DType *dataset_;          // data set
    const u64   *datapos_;          // data position
    
    int   N_;                       // total number of data points
    int   avg_d_;                   // average dimension of sparse data
    int   *labels_;                 // cluster labels O(n)
    
    std::vector<int> bktset_;       // bucket set
    std::vector<u64> bktpos_;       // bucket position
    std::vector<int> binset_;       // bin set
    std::vector<u64> binpos_;       // bin position
    std::vector<int> seedset_;      // seed set
    std::vector<u64> seedpos_;      // seed position
    
    // -------------------------------------------------------------------------
    void free();                    // free space for local parameters
    
    // -------------------------------------------------------------------------
    double data_to_buckets(         // phase 1: data -> buckets
        int m,                          // #hash tables
        int h,                          // #concat hash func 
        int t);                         // threshold of #point IDs in a bucket
    
    // -------------------------------------------------------------------------
    double buckets_to_bins(         // phase 2: buckets -> bins
        int  l,                         // #hash tables in minhash
        int  k,                         // #concat hash func in minhash
        int  delta,                     // threshold of #point IDs in a bin
        bool filter);                   // whether filter #point IDs in a bin
    
    // -------------------------------------------------------------------------
    double deduplicate(             // phase 2: remove duplicate bins
        int  l,                         // #hash tables in minhash
        int  k,                         // #concat hash func in minhash
        int  delta,                     // threshold of #point IDs in a bin
        bool filter);                   // whether filter #point IDs in a bin
        
    // -------------------------------------------------------------------------
    double bins_to_labels();        // phase 3: bins -> seeds -> labels
    
    // -------------------------------------------------------------------------
    void post_evaluation();         // post evaluation: labels -> seeds
};

// -----------------------------------------------------------------------------
template<class DType>
SILK<DType>::SILK(                  // constructor
    int   n,                            // number of data points
    int   d,                            // data dimension
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   b,                            // threshold of #bucket IDs in a bin
    float gbeta,                        // global beta
    float lbeta,                        // local  beta
    float galpha,                       // global alpha
    float lalpha,                       // local  alpha
    const char  *folder,                // output folder
    const DType *dataset,               // data set
    const u64   *datapos)               // data position
    : n_(n), d_(d), rank_(rank), size_(size), b_(b), gbeta_(gbeta), lbeta_(lbeta),
    galpha_(galpha), lalpha_(lalpha), dataset_(dataset), datapos_(datapos)
{
    assert((u64) n*size < MAX_INT); // N_ still range in int
    srand(RANDOM_SEED); // fix a random seed
    
    N_ = n * size;
    strncpy(folder_, folder, sizeof(folder_));
    labels_ = new int[n_];
    
    // calc avg_d: all coordinates divided by N_
    u64 all_coords = get_total_coords(size_, datapos[n]);
    avg_d_ = (int) ceil((double) all_coords / (double) N_);
}

// -----------------------------------------------------------------------------
template<class DType>
SILK<DType>::~SILK()                // destructor
{
    free();
    delete[] labels_;
}

// -----------------------------------------------------------------------------
template<class DType>
void SILK<DType>::free()            // free space for local parameters
{
    std::vector<int>().swap(bktset_);
    std::vector<u64>().swap(bktpos_);
    std::vector<int>().swap(binset_);
    std::vector<u64>().swap(binpos_);
    
    std::vector<int>().swap(seedset_);
    std::vector<u64>().swap(seedpos_);
}

// -----------------------------------------------------------------------------
template<class DType>
void SILK<DType>::display()         // display parameters
{
    printf("The parameters of SILK:\n");
    printf("n      = %d\n",   n_);
    printf("d      = %d\n",   d_);
    printf("rank   = %d\n",   rank_);
    printf("size   = %d\n",   size_);
    printf("N      = %d\n",   N_);
    printf("avg_d  = %d\n",   avg_d_);
    printf("b      = %d\n",   b_);
    printf("gbeta  = %.1f\n", gbeta_);
    printf("lbeta  = %.1f\n", lbeta_);
    printf("galpha = %.1f\n", galpha_);
    printf("lalpha = %.1f\n", lalpha_);
    printf("folder = %s\n\n", folder_);
}

// -----------------------------------------------------------------------------
void output_labels(                 // output labels to disk
    int   rank,                         // mpi rank
    int   n,                            // number of labels
    const int  *labels,                 // labels
    const char *prefix)                 // prefix
{
    // output binary format
    char fname[200]; sprintf(fname, "%s_labels_%d.bin", prefix, rank);
    FILE *fp = fopen(fname, "wb");
    if (!fp) { printf("Could not open %s\n", fname); exit(1); }

    fwrite(labels, sizeof(int), n, fp);
    fclose(fp);

    // also output text format for easy reading
    sprintf(fname, "%s_labels_%d.txt", prefix, rank);
    fp = fopen(fname, "w");
    if (!fp) { printf("Could not open %s\n", fname); exit(1); }
    
    for (int i = 0; i < n; ++i) fprintf(fp, "%d\n", labels[i]);
    fclose(fp);
}

// -----------------------------------------------------------------------------
template<class DType>
int SILK<DType>::clustering(        // generic distributed clustering
    int   m,                            // #hash tables (1st level minhash)
    int   h,                            // #concat hash func (1st level minhash)
    int   t,                            // threshold of #point IDs in a bucket
    int   l,                            // #hash tables (2nd level minhash)
    int   k,                            // #concat hash func (2nd level minhash)
    int   delta)                        // threshold of #point IDs in a bin
{
    clock_t start_cpu_time = clock();
    double  start_wc_time  = omp_get_wtime();
    
    // declare parameters & estimate hash cost
    char prefix[200];
    sprintf(prefix, "%ssilk_%d_%d_%d_%d_%d_%d", folder_, m, h, t, l, k, delta);
    printf("Params: m=%d, h=%d, t=%d, l=%d, k=%d, delta=%d, prefix=%s\n\n", 
        m, h, t, l, k, delta, prefix);
    
    // phase 1: 1st level minhash
    g_phase1_wc_time = data_to_buckets(m, h, t);
    if (g_tot_buckets == 0 || g_num_buckets == 0) return 1;

    // phase 2: 2nd level minhash + frequent data determination + deduplication
    double silk_time  = buckets_to_bins(l, k, delta, false);
    double dedup_time = deduplicate(1, 2, delta, true);
    g_phase2_wc_time = silk_time + dedup_time;
    if (g_tot_seeds == 0) return 1;
    
    // phase 3: initial seeds computation + data assignment
    g_phase3_wc_time = bins_to_labels();
    
    g_silk_cpu_time = (double) (clock() - start_cpu_time) / CLOCKS_PER_SEC;
    g_silk_wc_time  = omp_get_wtime() - start_wc_time;
    
    // post evaluation: re-compute centers & calc statistics
    post_evaluation();
#ifdef DEBUG_INFO
    // output_labels(rank_, n_, labels_, prefix);
#endif
    free();
    
    g_tot_cpu_time = (double) (clock() - start_cpu_time) / CLOCKS_PER_SEC;
    g_tot_wc_time  = omp_get_wtime() - start_wc_time;
    
    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
double SILK<DType>::data_to_buckets(// phase 1: data -> buckets
    int m,                              // #hash tables
    int h,                              // #concat hash func
    int t)                              // threshold of #point IDs in a bucket
{
    double start_time = omp_get_wtime();
    
    // minhash: data -> local hash results
    int *hash_results = new int[(u64) m*n_];
    int n_prime = 120001037 > N_ ? 120001037 : 1645333507;
    int d_prime = 120001037 > d_ ? 120001037 : 1645333507;
    printf("Rank #%d: n_prime=%d, d_prime=%d\n", rank_, n_prime, d_prime);
    
    minhash<DType>(rank_, n_, n_prime, d_prime, m, h, dataset_, datapos_, 
        hash_results);
    if (size_ > 1) {
        // shift local hash results in the distributed setting
        hash_results = shift_hash_results(size_, n_, m, hash_results);
    }
    double hash_time = omp_get_wtime() - start_time;
    
    // hash_results_to_buckets: local hash results -> local buckets
    g_num_buckets = hash_results_to_buckets(rank_, size_, n_, m, t, hash_results,
        bktset_, bktpos_);
    
    // get total number of buckets
    g_tot_buckets = get_total_buckets(size_, g_num_buckets);
    delete[] hash_results;
    double trans_time = omp_get_wtime() - start_time;
    
#ifdef DEBUG_INFO
    printf("Rank #%d: #Total Buckets=%d, #Local Buckets=%d, #IDs=%lu\n", rank_, 
        g_tot_buckets, g_num_buckets, bktpos_[g_num_buckets]);
    printf("Rank #%d: Data Transformation = %.2lf + %.2lf = %.2lf Seconds\n\n", 
        rank_, hash_time, trans_time - hash_time, trans_time);
#endif
    return trans_time;
}

// -----------------------------------------------------------------------------
template<class DType>
double SILK<DType>::buckets_to_bins(// phase 2: buckets -> bins
    int  l,                             // #hash tables in minhash
    int  k,                             // #concat hash func in minhash
    int  delta,                         // threshold of #point IDs in a bin
    bool filter)                        // whether filter #point IDs in a bin
{
    double start_time = omp_get_wtime();
    
    // minhash: local buckets -> local signatures
    int *signatures = new int[(u64) g_num_buckets*l];
    int n_prime = 120001037 > g_tot_buckets ? 120001037 : 1645333507;
    int d_prime = 120001037 > N_ ? 120001037 : 1645333507;
    printf("Rank #%d: n_prime=%d, d_prime=%d\n", rank_, n_prime, d_prime);
    
    minhash<int>(rank_, g_num_buckets, n_prime, d_prime, l, k, bktset_.data(), 
        bktpos_.data(), signatures);
    double minhash_time = omp_get_wtime() - start_time;

    // signatures_to_bins: local signatures -> global bins
    g_tot_bins = signatures_to_bins(rank_, size_, N_, g_num_buckets, l, b_, 
        delta, filter, gbeta_, lbeta_, bktset_.data(), bktpos_.data(), 
        signatures, binset_, binpos_);
    if (filter) g_tot_seeds = g_tot_bins;
    delete[] signatures;
    double silk_time = omp_get_wtime() - start_time;
    
#ifdef DEBUG_INFO
    printf("Rank #%d: #Total Bins=%d, #IDs=%lu\n", rank_, g_tot_bins, 
        binpos_[g_tot_bins]);
    printf("Rank #%d: SILK = %.2lf + %.2lf = %.2lf Seconds\n\n", rank_, 
        minhash_time, silk_time - minhash_time, silk_time);
#endif
    return silk_time;
}

// -----------------------------------------------------------------------------
template<class DType>
double SILK<DType>::deduplicate(    // phase 2: remove duplicate bins
    int  l,                             // #hash tables in minhash
    int  k,                             // #concat hash func in minhash
    int  delta,                         // threshold of #point IDs in a bin
    bool filter)                        // whether filter #point IDs in a bin
{
    double start_time = omp_get_wtime();
    
    // bins_to_buckets：global bins -> local buckets (for deduplication)
    int tot_buckets = g_tot_bins;
    int num_buckets = bins_to_buckets(rank_, size_, tot_buckets, binset_, 
        binpos_, bktset_, bktpos_);
    
    // minhash: local buckets -> local signatures
    int *signatures = new int[(u64) num_buckets*l];
    int n_prime = 120001037 > tot_buckets ? 120001037 : 1645333507;
    int d_prime = 120001037 > N_ ? 120001037 : 1645333507;
    printf("Rank #%d: n_prime=%d, d_prime=%d\n", rank_, n_prime, d_prime);
    
    minhash<int>(rank_, num_buckets, n_prime, d_prime, l, k, bktset_.data(), 
        bktpos_.data(), signatures);
    double minhash_time = omp_get_wtime() - start_time;
    
    // signatures_to_bins: local signatures -> global bins
    g_tot_seeds = signatures_to_bins(rank_, size_, N_, num_buckets, l, 0, delta, 
        filter, gbeta_, lbeta_, bktset_.data(), bktpos_.data(), signatures, 
        binset_, binpos_);
    delete[] signatures;
    double dedup_time = omp_get_wtime() - start_time;
    
#ifdef DEBUG_INFO
    printf("Rank #%d: #Total Seeds=%d, #IDs=%d\n", rank_, g_tot_seeds, 
        binpos_[g_tot_seeds]);
    printf("Rank #%d: Deduplicate = %.2lf + %.2lf = %.2lf Seconds\n\n", rank_, 
        minhash_time, dedup_time - minhash_time, dedup_time);
#endif
    return dedup_time;
}

// -----------------------------------------------------------------------------
template<class DType>
double SILK<DType>::bins_to_labels()// phase 3: bins -> seeds -> labels
{
    double start_time = omp_get_wtime();
    
    // bins_to_seeds: global bins -> global seeds
    bins_to_seeds<DType>(rank_, size_, n_, g_tot_seeds, avg_d_, galpha_,
        lalpha_, dataset_, datapos_, binset_.data(), binpos_.data(), 
        seedset_, seedpos_);
    double center_time = omp_get_wtime() - start_time;
    
    // data assignment: global seeds -> local labels
    approx_assign_data<DType>(rank_, n_, g_tot_seeds, dataset_, datapos_,
        seedset_.data(), seedpos_.data(), labels_);
    double assign_time = omp_get_wtime() - start_time;
    
#ifdef DEBUG_INFO
    printf("Rank #%d: Data Assignment = %.2lf + %.2lf = %.2lf Seconds\n\n", 
        rank_, center_time, assign_time - center_time, assign_time);
#endif
    return assign_time;
}

// -----------------------------------------------------------------------------
template<class DType>
void SILK<DType>::post_evaluation() // post evaluation: labels -> seeds
{
    clock_t start_cpu_time = clock();
    double  start_wc_time  = omp_get_wtime();
    
    // labels_to_bins: local labels -> global bins & re-number local labels
    g_k = labels_to_bins(rank_, size_, n_, g_tot_seeds, labels_, binset_, binpos_);
    
    // bins_to_seeds: global bins -> global seeds
    bins_to_seeds<DType>(rank_, size_, n_, g_k, avg_d_, galpha_, lalpha_, 
        dataset_, datapos_, binset_.data(), binpos_.data(), seedset_, seedpos_);
    
    double center_cpu_time = (double) (clock() - start_cpu_time) / CLOCKS_PER_SEC;
    double center_wc_time  = omp_get_wtime() - start_wc_time;
    
    // calc the statistics: mae and mse
    f32 *c_mae = new f32[g_k]; memset(c_mae, 0.0f, sizeof(f32)*g_k);
    f32 *c_mse = new f32[g_k]; memset(c_mse, 0.0f, sizeof(f32)*g_k);
    
    calc_local_stat_by_seeds<DType>(n_, g_k, labels_, dataset_, datapos_,
        seedset_.data(), seedpos_.data(), c_mae, c_mse);
    
    calc_global_stat(size_, n_, g_k, c_mae, c_mse, g_mae, g_mse);
    
    delete[] c_mae;
    delete[] c_mse;
    
    g_eval_cpu_time = (double) (clock() - start_cpu_time) / CLOCKS_PER_SEC;
    g_eval_wc_time  = omp_get_wtime() - start_wc_time;
    
#ifdef DEBUG_INFO
    printf("Rank #%d: k=%d, Evaluation = %.2lf + %.2lf = %.2lf Seconds\n\n", rank_, 
        g_k, center_wc_time, g_eval_wc_time - center_wc_time, g_eval_wc_time);
#endif
}

} // end namespace clustering
