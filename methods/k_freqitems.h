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
#include "bin.h"
#include "assign.cuh"
#include "eval.h"
#include "seeding.cuh"

namespace clustering {

// -----------------------------------------------------------------------------
//  KFreqItems: k-freqitems clustering for sparse data over Jaccard distance
// -----------------------------------------------------------------------------
template<class DType>
class KFreqItems {
public:
    KFreqItems(                     // constructor
        int   n,                        // number of data points
        int   d,                        // data dimension
        int   rank,                     // MPI rank
        int   size,                     // number of MPIs (size)
        int   max_iter,                 // maximum iteration
        float galpha,                   // global alpha
        float lalpha,                   // local  alpha
        const char  *folder,            // output folder
        const DType *dataset,           // data set
        const u64   *datapos);          // data position
    
    // -------------------------------------------------------------------------
    ~KFreqItems();                      // destructor
    
    // -------------------------------------------------------------------------
    void display();                 // display parameters
    
    // -------------------------------------------------------------------------
    int clustering(                 // k-freqitems clustering
        int   k,                        // #clusters (specified by users)
        int   seeding_algo,             // 0-Random 1-kmeans++ 2-kmeans|| 3-silk
        int   m1,                       // #hash tables (1st level minhash)
        int   h1,                       // #concat hash func (1st level minhash)
        int   t,                        // threshold of #point IDs in a bucket
        int   m2,                       // #hash tables (2nd level minhash)
        int   h2,                       // #concat hash func (2nd level minhash)
        int   b,                        // threshold of #bucket IDs in a bin
        int   delta,                    // threshold of #point IDs in a bin
        float gbeta,                    // global beta
        float lbeta);                   // local  beta
    
protected:
    int   n_;                       // number of data points
    int   d_;                       // data dimension
    int   rank_;                    // MPI rank
    int   size_;                    // number of MPIs (size)
    int   max_iter_;                // maximum iteration
    float galpha_;                  // global \alpha
    float lalpha_;                  // local  \alpha
    const DType *dataset_;          // data set
    const u64   *datapos_;          // data position
    char  folder_[200];             // output folder
    
    int   N_;                       // total number of data points
    int   avg_d_;                   // average dimension of sparse data
    int   *labels_;                 // cluster labels
    std::vector<int> binset_;       // bin set
    std::vector<u64> binpos_;       // bin position
    std::vector<int> seedset_;      // seed set
    std::vector<u64> seedpos_;      // seed position
    
    // -------------------------------------------------------------------------
    void free();                    // free space for local parameters
    
    // -------------------------------------------------------------------------
    void evaluate(                  // calc statistics
        int   K,                        // actual number of clusters
        float &mae,                     // mae (return)
        float &mse);                    // mse (return)
};

// -----------------------------------------------------------------------------
template<class DType>
KFreqItems<DType>::KFreqItems(      // constructor
    int   n,                            // number of data points
    int   d,                            // data dimension
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   max_iter,                     // maximum iteration
    float galpha,                       // global alpha
    float lalpha,                       // local  alpha
    const char  *folder,                // output folder
    const DType *dataset,               // data set
    const u64   *datapos)               // data position
    : n_(n), d_(d), rank_(rank), size_(size), N_(n*size), max_iter_(max_iter), 
    galpha_(galpha), lalpha_(lalpha), dataset_(dataset), datapos_(datapos)
{
    assert((u64) n*size < MAX_INT); // N_ still range in int
    
    srand(RANDOM_SEED); // fix a random seed
    strncpy(folder_, folder, sizeof(folder_)); // init folder_
    labels_ = new int[n]; // init label_
    
    // calc avg_d: all coordinates divided by N_
    u64 all_coords = get_total_coords(size_, datapos[n]);
    avg_d_ = (int) ceil((double) all_coords / (double) N_);
}

// -----------------------------------------------------------------------------
template<class DType>
KFreqItems<DType>::~KFreqItems()    // destructor
{
    free();
    delete[] labels_;
}

// -----------------------------------------------------------------------------
template<class DType>
void KFreqItems<DType>::free()      // free space for local parameters
{
    std::vector<int>().swap(binset_);
    std::vector<u64>().swap(binpos_);
    std::vector<int>().swap(seedset_);
    std::vector<u64>().swap(seedpos_);
}

// -----------------------------------------------------------------------------
template<class DType>
void KFreqItems<DType>::display()   // display parameters
{
    printf("The parameters of KFreqItems:\n");
    printf("n        = %d\n",   n_);
    printf("d        = %d\n",   d_);
    printf("rank     = %d\n",   rank_);
    printf("size     = %d\n",   size_);
    printf("N        = %d\n",   N_);
    printf("avg_d    = %d\n",   avg_d_);
    printf("max_iter = %d\n",   max_iter_);
    printf("galpha   = %.1f\n", galpha_);
    printf("lalpha   = %.1f\n", lalpha_);
    printf("folder   = %s\n\n", folder_);
}

// -----------------------------------------------------------------------------
void output_distinct_ids(           // output k distinct ids for as seeds
    int   rank,                         // MPI rank
    int   k,                            // number of seeds
    const int *distinct_ids,            // distinct ids
    const char *mname,                  // method name
    const char *folder)                 // output folder
{
    if (rank > 0) return;
    
    char fname[200]; sprintf(fname, "%s%d_%s_seeds.csv", folder, k, mname);
    FILE *fp = fopen(fname, "w");
    if (!fp) { printf("Could not open %s\n", fname); exit(1); }
    
    fprintf(fp, "k=%d\n", k);
    for (int i = 0; i < k; ++i) fprintf(fp, "%d\n", distinct_ids[i]);
    fclose(fp);
}

// -----------------------------------------------------------------------------
void output_centers(                // output k centers as seeds
    int   rank,                         // MPI rank
    int   k,                            // number of seeds
    const std::vector<int> &seedset,    // seed set (return)
    const std::vector<u64> &seedpos,    // seed position (return)
    const char *mname,                  // method name
    const char *folder)                 // output folder
{
    if (rank > 0) return;
    
    assert(seedpos.size()-1 == k);
    
    char fname[100]; sprintf(fname, "%s%d_%s_seeds.bin", folder, k, mname);
    FILE *fp = fopen(fname, "wb");
    if (!fp) { printf("Could not open %s\n", fname); exit(1); }

    fwrite(&k, sizeof(int), 1, fp);
    fwrite(seedpos.data(), sizeof(u64), k+1, fp);
    fwrite(seedset.data(), sizeof(int), seedpos[k], fp);
    fclose(fp);
}

// -----------------------------------------------------------------------------
void output_iter_info(              // output info for each k-freqitems iteration
    int    seeding_algo,                // seeding algorithm (0-3)
    int    k,                           // specified number of clusters
    int    iter,                        // which iteration
    int    max_iter,                    // maximum iteration
    int    K,                           // actual number of clusters
    float  mae,                         // mean absolute error
    float  mse,                         // mean square error
    double assign_cpu_time,             // data assignment cpu time
    double assign_wc_time,              // data assignment wall clock time
    double update_cpu_time,             // seed update cpu time
    double update_wc_time,              // seed update wall clock time
    double total_cpu_time,              // total cpu time so far
    double total_wc_time,               // total wall clock time so far
    const  char *folder)                // output folder
{
    // output binary format
    char fname[200]; 
    if (seeding_algo == 3) sprintf(fname, "%s%d_iter_silk.csv", folder, k);
    else sprintf(fname, "%s%d_iter_info.csv", folder, k);
    
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Could not open %s\n", fname); exit(1); }
    
    fprintf(fp, "%d,%f,%f,%.2lf,%.2lf,", K, mse, mae, total_wc_time, 
        total_cpu_time);
    fprintf(fp, "%d,%d,%d,%.2lf+%.2lf=%.2lf,%.2lf+%.2lf=%.2lf,%d\n", k, 
        max_iter, iter,  assign_wc_time,  update_wc_time  - assign_wc_time, 
        update_wc_time,  assign_cpu_time, update_cpu_time - assign_cpu_time, 
        update_cpu_time, seeding_algo);
    fclose(fp);
}

// -----------------------------------------------------------------------------
void output_labels(                 // output labels
    int   rank,                         // MPI rank
    int   n,                            // number of data points (local)
    int   k,                            // number of clusters
    int   seeding_algo,                 // seeding algorithm [0-3]
    int   m1,                           // #hash tables (1st level minhash)
    int   h1,                           // #concat hash func (1st level minhash)
    int   t,                            // threshold of #point IDs in a bucket
    int   m2,                           // #hash tables (2nd level minhash)
    int   h2,                           // #concat hash func (2nd level minhash)
    int   delta,                        // threshold of #point IDs in a bin
    const int *labels,                  // cluster labels [0,k-1]
    const char *folder)                 // output folder
{
    // get prefix
    char prefix[200];
    if (seeding_algo == 0) {
        sprintf(prefix, "%s%d_random",   folder, k);
    }
    else if (seeding_algo == 1) {
        sprintf(prefix, "%s%d_kmeanspp", folder, k);
    }
    else if (seeding_algo == 2) {
        sprintf(prefix, "%s%d_kmeansll", folder, k);
    }
    else if (seeding_algo == 3) {
        sprintf(prefix, "%s%d_silk_%d_%d_%d_%d_%d_%d", folder, k, m1, h1, t, 
            m2, h2, delta);
    }
    else {
        exit(1);
    }
    
    // output labels
    char fname[200]; sprintf(fname, "%s_%d.labels", prefix, rank);
    FILE *fp = fopen(fname, "wb");
    if (!fp) { printf("Could not open %s\n", fname); exit(1); }
    fwrite(labels, sizeof(int), n, fp);
    fclose(fp);
    
    // FILE *fp = fopen(fname, "w");
    // if (!fp) { printf("Could not open %s\n", fname); exit(1); }
    // for (int i = 0; i < n; ++i) fprintf(fp, "%d\n", labels[i]);
    // fclose(fp);
}

// -----------------------------------------------------------------------------
template<class DType>
int KFreqItems<DType>::clustering(  // k-freqitems clustering
    int   k,                            // #clusters (specified by users)
    int   seeding_algo,                 // 0-random 1-kmeans++ 2-kmeans|| 3-silk
    int   m1,                           // #hash tables (1st level minhash)
    int   h1,                           // #concat hash func (1st level minhash)
    int   t,                            // threshold of #point IDs in a bucket
    int   m2,                           // #hash tables (2nd level minhash)
    int   h2,                           // #concat hash func (2nd level minhash)
    int   b,                            // threshold of #bucket IDs in a bin
    int   delta,                        // threshold of #point IDs in a bin
    float gbeta,                        // global beta
    float lbeta)                        // local  beta
{
    clock_t start_cpu_time = clock();
    double  start_wc_time  = omp_get_wtime();
    
    // -------------------------------------------------------------------------
    //  initial seeding: select k data points as seeds
    // -------------------------------------------------------------------------
    int *distinct_ids = new int[k];
    int hard_device = 1; // 0-OpenMP; 1-GPUs
    if (seeding_algo == 0) {
        // randomized seeding
        random_seeding<DType>(rank_, size_, n_, N_, avg_d_, k, dataset_, 
            datapos_, distinct_ids, seedset_, seedpos_);
        // output_distinct_ids(rank_, k, distinct_ids, "random", folder_);
    }
    else if (seeding_algo == 1) {
        // k-means++ seeding (use OpemMP by default)
        int *weights = new int[N_]; memset(weights, 1, sizeof(int)*N_);
        kmeanspp_seeding<DType>(rank_, size_, n_, N_, avg_d_, k, 0,
            dataset_, datapos_, weights, distinct_ids, seedset_, seedpos_);
        // output_distinct_ids(rank_, k, distinct_ids, "kmeanspp", folder_);
        delete[] weights;
    }
    else if (seeding_algo == 2) {
        // k-means|| seeding (use GPUs by default)
        int l = k;
        int t = 5;
        kmeansll_seeding<DType>(rank_, size_, n_, N_, avg_d_, k, l, t, 1, 
            dataset_, datapos_, distinct_ids, seedset_, seedpos_);
        // output_distinct_ids(rank_, k, distinct_ids, "kmeansll", folder_);
    }
    else if (seeding_algo == 3) {
        // silk seeding (use GPUs by default)
        k = silk_seeding<DType>(rank_, size_, n_, N_, d_, avg_d_, k, m1, h1, 
            t, m2, h2, b, delta, gbeta, lbeta, galpha_, lalpha_, dataset_, 
            datapos_, seedset_, seedpos_);
        // output_centers(rank_, k, seedset_, seedpos_, "silk", folder_);
    }
    else {
        exit(1);
    }
    delete[] distinct_ids;
    g_init_cpu_time = (double) (clock() - start_cpu_time) / CLOCKS_PER_SEC;
    g_init_wc_time  = omp_get_wtime() - start_wc_time;
    
#ifdef DEBUG_INFO
    if (rank_ == 0) {
        printf("\nRank #%d: k=%d, init_time=%.2lf seconds\n\n", rank_, k, 
            g_init_wc_time);
    }
#endif

    // -------------------------------------------------------------------------
    //  assignment-update iterations
    // -------------------------------------------------------------------------
    int    K = k; // actual number of clusters (K <= k)
    float  mae = -1.0f, mse = -1.0f;
    double assign_cpu_time, assign_wc_time, update_cpu_time, update_wc_time;
    
    g_mse = MAX_FLOAT;
    for (int iter = 1; iter <= max_iter_; ++iter) {
        clock_t local_start_ctime = clock();
        double  local_start_wtime = omp_get_wtime();
        
        // data assignment (assign.cu)
        exact_assign_data<DType>(rank_, n_, K, dataset_, datapos_, 
            seedset_.data(), seedpos_.data(), labels_);
        
        assign_cpu_time = (double) (clock()-local_start_ctime) / CLOCKS_PER_SEC;
        assign_wc_time  = omp_get_wtime() - local_start_wtime;
        
        // update freqitems & re-number the labels in [0,K-1] (bin.cu)
        K = labels_to_bins(rank_, size_, n_, K, labels_, binset_, binpos_);
        
        // convert bins into seeds (assign.cuh)
        bins_to_seeds<DType>(rank_, size_, n_, K, avg_d_, galpha_, lalpha_, 
            dataset_, datapos_, binset_.data(), binpos_.data(), seedset_, 
            seedpos_);
        
        // evaluation based new freqitems and new labels
        evaluate(K, mae, mse);
        
        update_cpu_time = (double) (clock()-local_start_ctime) / CLOCKS_PER_SEC;
        update_wc_time  = omp_get_wtime() - local_start_wtime;
        g_tot_cpu_time  = (double) (clock()-start_cpu_time) / CLOCKS_PER_SEC;
        g_tot_wc_time   = omp_get_wtime() - start_wc_time;
        
        if (mse < g_mse) {
            g_k = K; g_mae = mae; g_mse = mse; g_iter = iter;
            g_kfreqitems_cpu_time = g_tot_cpu_time;
            g_kfreqitems_wc_time  = g_tot_wc_time;
        }
        
#ifdef DEBUG_INFO
        if (rank_ == 0) {
            printf("Rank #%d: iter=%d/%d, k=%d, mse=%f, mae=%f, "
                "time=%.2lf+%.2lf=%.2lf, total_time=%.2lf\n\n", rank_, 
                iter, max_iter_, K, mse, mae, assign_wc_time, 
                update_wc_time-assign_wc_time, update_wc_time, g_tot_wc_time);
            
            output_iter_info(seeding_algo, k, iter, max_iter_, K, mae, mse, 
                assign_cpu_time, assign_wc_time, update_cpu_time, 
                update_wc_time, g_tot_cpu_time, g_tot_wc_time, folder_);
        }
#endif
    }
#ifdef DEBUG_INFO
    // output_labels(rank_, n_, k, seeding_algo, m1, h1, t, m2, h2, delta, 
    //     labels_, folder_);
    // output_centers(rank_, k, seedset_, seedpos_, "silk", folder_);
#endif
    free();
    
    g_tot_cpu_time  = (double) (clock()-start_cpu_time) / CLOCKS_PER_SEC;
    g_tot_wc_time   = omp_get_wtime() - start_wc_time;
    g_iter_cpu_time = (g_tot_cpu_time - g_init_cpu_time) / max_iter_;
    g_iter_wc_time  = (g_tot_wc_time  - g_init_wc_time)  / max_iter_;
    
    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
void KFreqItems<DType>::evaluate(   // update k freqitems & calc statistics
    int   K,                            // actual number of clusters
    float &mae,                         // mae (return)
    float &mse)                         // mse (return)
{
    // calc mae and mse for clusters
    f32 *c_mae = new f32[K]; memset(c_mae, 0.0f, sizeof(f32)*K);
    f32 *c_mse = new f32[K]; memset(c_mse, 0.0f, sizeof(f32)*K);
    
    calc_local_stat_by_seeds<DType>(n_, K, labels_, dataset_, datapos_,
        seedset_.data(), seedpos_.data(), c_mae, c_mse);
    
    calc_global_stat(size_, n_, K, c_mae, c_mse, mae, mse);
    
    delete[] c_mae;
    delete[] c_mse;
}

} // end namespace clustering
