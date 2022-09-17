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
void generate_k_distinct_ids(       // generate k distinct ids
    int k,                              // k value
    int n,                              // total range
    int *distinct_ids);                 // distinct ids (return)

// -----------------------------------------------------------------------------
template<class DType>
void distinct_ids_to_local_seed(    // get local seeds from distinct ids
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   avg_d,                        // average dimension of data points
    int   k,                            // top-k value
    const int   *distinct_ids,          // distinct ids
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    std::vector<int> &local_seedset,    // local seedset (return)
    std::vector<u64> &local_seedpos)    // local seedpos (return)
{
    local_seedset.reserve(k*avg_d); // estimate, not correct
    local_seedpos.reserve(k+1);     // estimate, not correct
    local_seedpos.push_back(0);
    
    int lower_bound = n * rank;
    int upper_bound = n + lower_bound;
    int id = -1, len = 0;
    for (int i = 0; i < k; ++i) {
        id = distinct_ids[i];
        if (id < lower_bound || id >= upper_bound) continue;
        
        // get the local data (by id)
        id -= lower_bound;
        const DType *data = &dataset[datapos[id]];
        len = get_length(id, datapos);
        
        // add this data to local seedset and seedpos
        local_seedset.insert(local_seedset.end(), data, data+len);
        local_seedpos.push_back(len);
    }
}

// -----------------------------------------------------------------------------
void gather_all_local_seedset(      // gather all local seedsets to root
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    const std::vector<int> &local_seedset, // local seedset
    std::vector<int> &seedset);         // seedset at root (return)

// -----------------------------------------------------------------------------
void gather_all_local_seedpos(      // gather all local seedpos to root
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   k,                            // k value
    const std::vector<u64> &local_seedpos, // local seedpos
    std::vector<u64> &seedpos);         // seedpos at root (return)

// -----------------------------------------------------------------------------
template<class DType>
void get_k_seeds(                   // get k seeds based on distinct ids
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   avg_d,                        // average dimension of sparse data
    int   k,                            // number of seeds
    const int   *distinct_ids,          // k distinct ids
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    std::vector<int> &seedset,          // seed set (return)
    std::vector<u64> &seedpos)          // seed position (return)
{
    std::vector<int>().swap(seedset);
    std::vector<u64>().swap(seedpos);
    
    // -------------------------------------------------------------------------
    //  get local seedset and seedpos
    // -------------------------------------------------------------------------
    std::vector<int> local_seedset;
    std::vector<u64> local_seedpos;
    
    distinct_ids_to_local_seed<DType>(rank, size, n, avg_d, k, distinct_ids,
        dataset, datapos, local_seedset, local_seedpos);
    
    // -------------------------------------------------------------------------
    //  get global seedset and seedpos to root
    // -------------------------------------------------------------------------
    if (size == 1) {
        // single-thread case: directly swap local seed and seed
        seedset.swap(local_seedset);
        seedpos.swap(local_seedpos);
    }
    else {
        // gather all local seeds into seedset and seedpos to root
        gather_all_local_seedset(rank, size, local_seedset, seedset);
        gather_all_local_seedpos(rank, size, k, local_seedpos, seedpos);
    
        // broadcast global seedset and seedpos from root to other threads
        broadcast_set_and_pos(rank, size, seedset, seedpos);
        
        std::vector<int>().swap(local_seedset);
        std::vector<u64>().swap(local_seedpos);
    }
    // accumulate the length of seeds to get the start position of each seed
    for (int i = 1; i <= k; ++i) seedpos[i] += seedpos[i-1];
}

// -----------------------------------------------------------------------------
template<class DType>
void random_seeding(                // init k centers by random seeding
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   N,                            // total number of data points
    int   avg_d,                        // average dimension of sparse data
    int   k,                            // number of seeds
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    int   *distinct_ids,                // k distinct ids (return)
    std::vector<int> &seedset,          // seed set (return)
    std::vector<u64> &seedpos)          // seed position (return)
{
    srand(RANDOM_SEED); // fix a random seed
    
    // -------------------------------------------------------------------------
    //  generate k distinct ids from [0, N-1]
    // -------------------------------------------------------------------------
    if (rank == 0) generate_k_distinct_ids(k, N, distinct_ids);
    if (size > 1) {
        // multi-thread case: broadcast distinct_ids from root to other threads
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(distinct_ids, k, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // -------------------------------------------------------------------------
    //  get the global seedset and seedpos by the k distinct ids
    // -------------------------------------------------------------------------
    get_k_seeds<DType>(rank, size, n, avg_d, k, distinct_ids, dataset, datapos, 
        seedset, seedpos);
}

// -----------------------------------------------------------------------------
void broadcast_target_data(         // broadcast target data to all threads
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    std::vector<int> &target_data);     // target data (return)

// -----------------------------------------------------------------------------
template<class DType>
void get_data_by_id(                // get a data point by input id
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   id,                           // input data id
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    std::vector<int> &target_data)      // target data (return)
{
    std::vector<int>().swap(target_data);
    
    // get local target_data
    int lower_bound = n * rank;
    int upper_bound = n + lower_bound;
    if (lower_bound <= id && id < upper_bound) {
        // retrieve the target data
        id -= lower_bound;
        const DType *data = &dataset[datapos[id]];
        int len = get_length(id, datapos);
        
        target_data.reserve(len);
        target_data.insert(target_data.end(), data, data+len);
    }
    // if multi-thread case: broadcast the target_data to all threads
    if (size > 1) broadcast_target_data(rank, size, target_data);
}

// -----------------------------------------------------------------------------
template<class DType>
void update_nn_dist(                // update nn_dist for input data
    int   did,                          // input data id
    int   n_seed,                       // length of input seed
    const int   *seed,                  // input seed
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    float &nn_dist)                     // nn_dist (return)
{
    int n_data = get_length(did, datapos);
    const DType *data = &dataset[datapos[did]];
    
    float dist = jaccard_dist<DType>(n_data, n_seed, data, seed);
    if (nn_dist > dist) nn_dist = dist;
}

// -----------------------------------------------------------------------------
template<class DType>
void update_nn_dist(                // update nn_dist for local data by a seed
    int   n,                            // number of data points
    const std::vector<int> &seed,       // last seed
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    float *nn_dist)                     // nn_dist (return)
{
    int n_seed = (int) seed.size();
    const int *seed_ptr = (const int*) seed.data();
    
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        update_nn_dist<DType>(i, n_seed, seed_ptr, dataset, datapos, nn_dist[i]);
    }
}

// -----------------------------------------------------------------------------
template<class DType>
void update_nn_dist(                // update nn_dist for local data
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    const std::vector<int> &seed,       // last seed
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    float *nn_dist);                    // nn_dist (return)

// -----------------------------------------------------------------------------
template<class DType>
void update_dist_and_prob(          // update nn_dist and prob by last seed
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   N,                            // total number of data points
    int   hard_device,                  // used hard device: 0-OpenMP, 1-GPU
    const std::vector<int> &seed,       // last seed
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    const int   *weights,               // weights of data set
    float *nn_dist,                     // nn_dist (return)
    float *prob)                        // probability (return)
{
    // -------------------------------------------------------------------------
    //  update nn_dist for the local data
    // -------------------------------------------------------------------------
    if (hard_device == 0) {
        // use OpenMP to update nn_dist (by default)
        update_nn_dist<DType>(n, seed, dataset, datapos, nn_dist);
    }
    else {
        // use GPUs to update nn_dist
        update_nn_dist<DType>(rank, n, seed, dataset, datapos, nn_dist);
    }
    // -------------------------------------------------------------------------
    //  get global nn_dist to root
    // -------------------------------------------------------------------------
    float *all_nn_dist = new float[N];
    if (size == 1) {
        // single-thread case: directly copy one to another
        std::copy(nn_dist, nn_dist + n, all_nn_dist);
    }
    else {
        // multi-thread case: gather nn_dist to root
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gather(nn_dist, n, MPI_FLOAT, all_nn_dist, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    // -------------------------------------------------------------------------
    //  @root: update global probability array
    // -------------------------------------------------------------------------
    if (rank == 0) {
        prob[0] = weights[0] * SQR(all_nn_dist[0]);
        for (int i = 1; i < N; ++i) {
            prob[i] = prob[i-1] + weights[i] * SQR(all_nn_dist[i]);
        }
    }
    delete[] all_nn_dist;
}

// -----------------------------------------------------------------------------
template<class DType>
void kmeanspp_seeding(              // init k centers by k-means++
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   N,                            // total number of data points
    int   avg_d,                        // average dimension of sparse data
    int   k,                            // number of seeds
    int   hard_device,                  // used hard device: 0-OpenMP, 1-GPU
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    const int   *weights,               // weights of data set
    int   *distinct_ids,                // k distinct ids (return)
    std::vector<int> &seedset,          // seed set (return)
    std::vector<u64> &seedpos)          // seed position (return)
{
    srand(RANDOM_SEED); // fix a random seed
    
    // -------------------------------------------------------------------------
    //  init nn_dist array & probability array
    // -------------------------------------------------------------------------
    float *nn_dist = new float[n];
    for (int i = 0; i < n; ++i) nn_dist[i] = MAX_FLOAT;
        
    float *prob = new float[N];
    prob[0] = (float) weights[0];
    for (int i = 1; i < N; ++i) prob[i] = prob[i-1] + weights[i];
    
    // -------------------------------------------------------------------------
    //  sample the first center
    // -------------------------------------------------------------------------
    int id = -1;
    // @root: sample the 1st center uniformly at random
    if (rank == 0) {
        float val = uniform(0.0f, prob[N-1]);
        id = std::lower_bound(prob, prob+N, val) - prob;
    }
    // broadcast id from root to other threads if multi-thread case
    if (size > 1) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&id, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    distinct_ids[0] = id;
    
    // -------------------------------------------------------------------------
    //  sample the remaining (k-1) centers by D^2 sampling
    // -------------------------------------------------------------------------
    std::vector<int> last_seed;
    for (int i = 1; i < k; ++i) {
        // get the last_seed by id
        get_data_by_id<DType>(rank, size, n, id, dataset, datapos, last_seed);
        
        // update nn_dist and prob by last_seed
        update_dist_and_prob<DType>(rank, size, n, N, hard_device, last_seed, 
            dataset, datapos, weights, nn_dist, prob);
        
        // @root: sample the i-th center (id) by D^2 sampling
        if (rank == 0) {
            float val = uniform(0.0f, prob[N-1]);
            id = std::lower_bound(prob, prob+N, val) - prob;
        }
        // broadcast id from root to other threads if multi-thread case
        if (size > 1) {
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Bcast(&id, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        distinct_ids[i] = id;
        
#ifdef DEBUG_INFO
        if (rank==0 && (i+1)%100==0) printf("Rank #%d: %d/%d\n", rank, i+1, k);
#endif
    }
    // -------------------------------------------------------------------------
    //  get the global seedset and seedpos by the k distinct ids
    // -------------------------------------------------------------------------
    get_k_seeds<DType>(rank, size, n, avg_d, k, distinct_ids, dataset, datapos, 
        seedset, seedpos);

    // release space
    std::vector<int>().swap(last_seed);
    delete[] nn_dist;
    delete[] prob;
}

// -----------------------------------------------------------------------------
template<class DType>
void update_nn_dist(                // update nn_dist for input data
    int   did,                          // input data id
    int   k,                            // number of seeds
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    const int   *seedset,               // seedset
    const u64   *seedpos,               // seedpos
    float &nn_dist)                     // nn_dist (return)
{
    // retrieve this data by id
    int n_data = get_length(did, datapos);
    const DType *data = &dataset[datapos[did]];
    
    // compare with k seeds and update nn_dist
    for (int i = 0; i < k; ++i) {
        int   n_seed = get_length(i, seedpos);
        const int *seed = seedset + seedpos[i];
        
        float dist = jaccard_dist<DType>(n_data, n_seed, data, seed);
        if (nn_dist > dist) nn_dist = dist;
    }
}

// -----------------------------------------------------------------------------
template<class DType>
void update_nn_dist(                // update nn_dist for local data by a seed
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    const int   *seedset,               // seedset
    const u64   *seedpos,               // seedpos
    float *nn_dist)                     // nn_dist (return)
{
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        update_nn_dist<DType>(i, k, dataset, datapos, seedset, seedpos, nn_dist[i]);
    }
}

// -----------------------------------------------------------------------------
template<class DType>
void update_nn_dist(                // update nn_dist for local data by k seeds
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    const int   *seedset,               // seed set
    const u64   *seedpos,               // seed position
    float *nn_dist);                    // nn_dist (return)

// -----------------------------------------------------------------------------
template<class DType>
void update_dist_and_prob(          // update nn_dist & prob by seedset & seedpos
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   N,                            // total number of data points
    int   hard_device,                  // used hard device: 0-OpenMP, 1-GPU
    const std::vector<int> &seedset,    // seed set
    const std::vector<u64> &seedpos,    // seed position
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    float *nn_dist,                     // nn_dist (return)
    float *prob)                        // probability (return)
{
    // -------------------------------------------------------------------------
    //  update nn_dist for the local data
    // -------------------------------------------------------------------------
    int k = (int) seedpos.size() - 1;
    if (hard_device == 0) {
        // use OpenMP to update nn_dist
        update_nn_dist<DType>(n, k, dataset, datapos, seedset.data(), 
            seedpos.data(), nn_dist);
    }
    else {
        // use GPUs to update nn_dist (by default)
        update_nn_dist<DType>(rank, n, k, dataset, datapos, seedset.data(), 
            seedpos.data(), nn_dist);
    }
    // -------------------------------------------------------------------------
    //  get global nn_dist to root
    // -------------------------------------------------------------------------
    float *all_nn_dist = new float[N];
    if (size == 1) {
        // single-thread case: directly copy one to another
        assert(n == N);
        std::copy(nn_dist, nn_dist + n, all_nn_dist);
    }
    else {
        // multi-thread case: gather nn_dist to root
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gather(nn_dist, n, MPI_FLOAT, all_nn_dist, n, MPI_FLOAT, 0, 
            MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    // -------------------------------------------------------------------------
    //  @root: update global probability array to all threads
    // -------------------------------------------------------------------------
    if (rank == 0) {
        prob[0] = SQR(all_nn_dist[0]);
        for (int i = 1; i < N; ++i) prob[i] = prob[i-1] + SQR(all_nn_dist[i]);
    }
    delete[] all_nn_dist;
}

// -----------------------------------------------------------------------------
template<class DType>
void kmeansll_overseeding(          // get (tl+1) centers by k-means|| overseeding
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   N,                            // total number of data points
    int   avg_d,                        // average dimension of sparse data
    int   l,                            // oversampling factor
    int   t,                            // number of iterations
    int   hard_device,                  // used hard device: 0-OpenMP, 1-GPU
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    int   *all_distinct_ids,            // (t*l+1) distinct ids (return)
    std::vector<int> &seedset,          // seed set (return)
    std::vector<u64> &seedpos)          // seed position (return)
{
    srand(RANDOM_SEED); // fix a random seed

    // -------------------------------------------------------------------------
    //  init nn_dist array & probability array
    // -------------------------------------------------------------------------
    float *nn_dist = new float[n];
    for (int i = 0; i < n; ++i) nn_dist[i] = MAX_FLOAT;
        
    float *prob = new float[N];
    for (int i = 0; i < N; ++i) prob[i] = i;
    
    // -------------------------------------------------------------------------
    //  sample the first center
    // -------------------------------------------------------------------------
    int id = -1;
    // @root: sample the 1st center uniformly at random
    if (rank == 0) {
        float val = uniform(0.0f, prob[N-1]);
        id = std::lower_bound(prob, prob+N, val) - prob;
    }
    // broadcast id from root to other threads if multi-thread case
    if (size > 1) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&id, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    all_distinct_ids[0] = id;
    
    // -------------------------------------------------------------------------
    //  sample the remaining (t*l) centers in t iterations by D^2 sampling
    // -------------------------------------------------------------------------
    std::vector<int> last_seedset;
    std::vector<u64> last_seedpos;
    
    for (int i = 0; i < t; ++i) {
#ifdef DEBUG_INFO
        printf("Rank #%d: %d/%d\n", rank, i+1, t);
#endif
        // get the last l seeds
        if (i == 0) {
            get_data_by_id<DType>(rank, size, n, id, dataset, datapos, last_seedset);
            
            std::vector<u64>().swap(last_seedpos);
            last_seedpos.push_back(0);
            last_seedpos.push_back(last_seedset.size());
        }
        else {
            int *distinct_ids = &all_distinct_ids[(i-1)*l+1];
            
            get_k_seeds<DType>(rank, size, n, avg_d, l, distinct_ids, dataset, 
                datapos, last_seedset, last_seedpos);
        }
        // update nn_dist and prob by the last l seeds
        update_dist_and_prob<DType>(rank, size, n, N, hard_device, last_seedset, 
            last_seedpos, dataset, datapos, nn_dist, prob);
        
        // @root: sample l centers (distinct_ids) by D^2 sampling
        int *distinct_ids = &all_distinct_ids[i*l+1];
        if (rank == 0) {
            for (int j = 0; j < l; ++j) {
                float val = uniform(0.0f, prob[N-1]);
                distinct_ids[j] = std::lower_bound(prob, prob+N, val) - prob;
            }
        }
        // broadcast l distinct_ids from root to other threads if multi-thread case
        if (size > 1) {
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Bcast(distinct_ids, l, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    // -------------------------------------------------------------------------
    //  get the global seedset and seedpos by the (t*l+1) distinct ids
    // -------------------------------------------------------------------------
    get_k_seeds<DType>(rank, size, n, avg_d, t*l+1, all_distinct_ids, dataset, 
        datapos, seedset, seedpos);

    // release space
    std::vector<int>().swap(last_seedset);
    std::vector<u64>().swap(last_seedpos);
    delete[] nn_dist;
    delete[] prob;
}

// -----------------------------------------------------------------------------
void labels_to_weights(             // convert local labels to global weights
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const int *labels,                  // labels for n local data
    int   *weights);                    // weights for k seeds (return)

// -----------------------------------------------------------------------------
template<class DType>
void get_weights_for_seeds(         // get weights for k seeds
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    const int   *seedset,               // seed set
    const u64   *seedpos,               // seed position
    int   *weights)                     // weights for k seeds (return)
{
    // conduct data assginment to get the local labels
    int *labels = new int[n];
    exact_assign_data<DType>(rank, n, k, dataset, datapos, seedset, seedpos, 
        labels);
    
    // convert local labels into global weights
    labels_to_weights(rank, size, n, k, labels, weights);
    
    // release space
    delete[] labels;
}

// -----------------------------------------------------------------------------
template<class DType>
void kmeansll_seeding(              // init k centers by k-means||
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   N,                            // total number of data points
    int   avg_d,                        // average dimension of sparse data
    int   k,                            // number of seeds
    int   l,                            // oversampling factor
    int   t,                            // number of iterations
    int   hard_device,                  // used hard device: 0-OpenMP, 1-GPU
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    int   *distinct_ids,                // k distinct ids (return)
    std::vector<int> &seedset,          // seed set (return)
    std::vector<u64> &seedpos)          // seed position (return)
{
    // -------------------------------------------------------------------------
    //  k-means|| overseeding
    // -------------------------------------------------------------------------
#ifdef DEBUG_INFO
    printf("Rank #%d: k-means|| overseeding (t=%d, l=%d)\n\n", rank, t, l);
#endif
    int K = t * l + 1; // expected number of seeds for overseeding
    int *all_distinct_ids = new int[K];
    std::vector<int> over_seedset;
    std::vector<u64> over_seedpos;
    
    kmeansll_overseeding<DType>(rank, size, n, N, avg_d, l, t, hard_device,
        dataset, datapos, all_distinct_ids, over_seedset, over_seedpos);
    assert(over_seedpos.size()-1 == K);
    
    // -------------------------------------------------------------------------
    //  get weights for K seeds by data assignment
    // -------------------------------------------------------------------------
#ifdef DEBUG_INFO
    printf("Rank #%d: data assignment => weights (n=%d, K=%d)\n\n", rank, n, K);
#endif
    int *weights = new int[K];
    get_weights_for_seeds<DType>(rank, size, n, K, dataset, datapos, 
        over_seedset.data(), over_seedpos.data(), weights);
    
    // -------------------------------------------------------------------------
    //  call k-means++ for final refinement using a single thread (@root)
    // -------------------------------------------------------------------------
#ifdef DEBUG_INFO
    printf("Rank #%d: k-means++ refinement (k=%d)\n\n", rank, k);
#endif
    if (rank == 0) {
        // use OpenMP for k-means++ by default
        int *local_distinct_ids = new int[k];
        kmeanspp_seeding<int>(0, 1, K, K, avg_d, k, 0, over_seedset.data(),
            over_seedpos.data(), weights, local_distinct_ids, seedset, seedpos);
    
        // get distinct_ids from all_distinct_ids and local_distinct_ids
        for (int i = 0; i < k; ++i) {
            int pos = local_distinct_ids[i];
            distinct_ids[i] = all_distinct_ids[pos];
        }
        delete[] local_distinct_ids;
    }
    // broadcast distinct_ids, seedset and seedpos from root to other threads
    if (size > 1) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(distinct_ids, k, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        
        broadcast_set_and_pos(rank, size, seedset, seedpos);
    }
    // release space
    std::vector<int>().swap(over_seedset);
    std::vector<u64>().swap(over_seedpos);
    delete[] all_distinct_ids;
    delete[] weights;
}

// -----------------------------------------------------------------------------
template<class DType>
int silk_overseeding(               // silk overseeding
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   N,                            // total number of data points
    int   d,                            // real dimension of sparse data
    int   avg_d,                        // average dimension of sparse data
    int   m1,                           // #hash tables (1st level)
    int   h1,                           // #concat hash func (1st level)
    int   t,                            // threshold of #point IDs in a bucket
    int   m2,                           // #hash tables (2nd level)
    int   h2,                           // #concat hash func (2nd level)
    int   b,                            // threshold of #bucket IDs in a bin
    int   delta,                        // threshold of #point IDs in a bin
    float gbeta,                        // global beta
    float lbeta,                        // local  beta
    float galpha,                       // global alpha
    float lalpha,                       // local  alpha
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    std::vector<int> &seedset,          // seed set (return)
    std::vector<u64> &seedpos)          // seed position (return)
{
    // -------------------------------------------------------------------------
    //  phase 1: transform local sparse data to local buckets
    // -------------------------------------------------------------------------
    std::vector<int> bktset; // bucket set
    std::vector<u64> bktpos; // bucket position
    
    int n_prime = 120001037 > N ? 120001037 : 1645333507;
    int d_prime = 120001037 > d ? 120001037 : 1645333507;
    
    // using minhash to get local hash results
    int *hash_results = new int[(u64) m1*n];
    minhash<DType>(rank, n, n_prime, d_prime, m1, h1, dataset, datapos, 
        hash_results);
    if (size > 1) hash_results = shift_hash_results(size, n, m1, hash_results);
    
    // convert local hash results into local buckets
    int num_buckets = hash_results_to_buckets(rank, size, n, m1, t, hash_results, 
        bktset, bktpos);
    
    // get total number of buckets
    int tot_buckets = get_total_buckets(size, num_buckets);
    delete[] hash_results;
    
    printf("Rank #%d: num_buckets=%d, tot_buckets=%d\n\n", rank, num_buckets,
        tot_buckets);
    
    // -------------------------------------------------------------------------
    //  phase 2: transform local buckets to global bins
    // -------------------------------------------------------------------------
    std::vector<int> binset; // bin set
    std::vector<u64> binpos; // bin position
    
    n_prime = 120001037 > tot_buckets ? 120001037 : 1645333507;
    d_prime = 120001037 > N ? 120001037 : 1645333507;
    
    // minhash: convert local buckets into local signatures
    int *signatures = new int[(u64) num_buckets*m2];
    minhash<int>(rank, num_buckets, n_prime, d_prime, m2, h2, bktset.data(),
        bktpos.data(), signatures);
    
    // signatures_to_bins: convert local signatures into global bins
    int tot_bins = signatures_to_bins(rank, size, N, num_buckets, m2, b, 
        delta, false, gbeta, lbeta, bktset.data(), bktpos.data(), 
        signatures, binset, binpos);
    delete[] signatures; signatures = nullptr;

    printf("Rank #%d: tot_bins=%d\n\n", rank, tot_bins);
    
    // -------------------------------------------------------------------------
    //  phase 3: conduct minhash again to deduplicate global bins
    // -------------------------------------------------------------------------
    // convert global bins into local buckets
    tot_buckets = tot_bins;
    num_buckets = bins_to_buckets(rank, size, tot_buckets, binset, binpos, 
        bktset, bktpos);
    
    n_prime = 120001037 > tot_buckets ? 120001037 : 1645333507;
    d_prime = 120001037 > N ? 120001037 : 1645333507;
    
    // minhash: convert local buckets into local signatures
    m2 = 1; h2 = 2;
    signatures = new int[(u64) num_buckets*m2];
    minhash<int>(rank, num_buckets, n_prime, d_prime, m2, h2, bktset.data(), 
        bktpos.data(), signatures);
    
    // signatures_to_bins: convert local signatures into global bins
    tot_bins = signatures_to_bins(rank, size, N, num_buckets, m2, 0, 
        delta, true, gbeta, lbeta, bktset.data(), bktpos.data(), 
        signatures, binset, binpos);
    delete[] signatures;
    
    printf("Rank #%d: K=%d\n\n", rank, tot_bins);
    
    // -------------------------------------------------------------------------
    //  phase 4: transform bins to seeds
    // -------------------------------------------------------------------------
    bins_to_seeds<DType>(rank, size, n, tot_bins, avg_d, galpha, lalpha, 
        dataset, datapos, binset.data(), binpos.data(), seedset, seedpos);
    
    // clear space
    std::vector<int>().swap(bktset);
    std::vector<u64>().swap(bktpos);
    std::vector<int>().swap(binset);
    std::vector<u64>().swap(binpos);
    
    return tot_bins;
}

// -----------------------------------------------------------------------------
int early_stop(                     // early stop process
    std::vector<int> &over_seedset,     // over seed set (allow modify)
    std::vector<u64> &over_seedpos,     // over seed position (allow modify)
    std::vector<int> &seedset,          // seed set (return)
    std::vector<u64> &seedpos);         // seed position (return)
 
// -----------------------------------------------------------------------------
template<class DType>
void labels_to_weights(             // get weights from labels & update seeds
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   avg_d,                        // average dimension of sparse data
    int   k,                            // number of seeds
    float galpha,                       // global alpha
    float lalpha,                       // local  alpha
    const int   *labels,                // labels for n local data
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    std::vector<int> &seedset,          // seed set (return)
    std::vector<u64> &seedpos,          // seed position (return)
    int   *weights)                     // weights for k seeds (return)
{
    std::vector<int> binset; // bin set
    std::vector<u64> binpos; // bin position
    
    // convert local labels into global bins and re-number local labels
    int new_k = labels_to_index(size, n, k, labels, binset, binpos);
    assert(new_k <= k && new_k > 0);
    
    // update weights based on new_k bins
    for (int i = 0; i < new_k; ++i) {
        weights[i] = get_length(i, binpos.data());
    }
    
    // re-generate seeds based on global bins
    bins_to_seeds<DType>(rank, size, n, new_k, avg_d, galpha, lalpha, 
        dataset, datapos, binset.data(), binpos.data(), seedset, seedpos);
    
    std::vector<int>().swap(binset);
    std::vector<u64>().swap(binpos);
}

// -----------------------------------------------------------------------------
template<class DType>
int get_weights_for_seeds(          // get weights for k seeds
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   avg_d,                        // average dimension of sparse data
    int   k,                            // number of seeds
    float galpha,                       // global alpha
    float lalpha,                       // local  alpha
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    std::vector<int> &seedset,          // seed set (return)
    std::vector<u64> &seedpos,          // seed position (return)
    int   *weights)                     // weights for k seeds (return)
{
    // conduct data assginment to get the local labels
    int *labels = new int[n];
    approx_assign_data<DType>(rank, n, k, dataset, datapos, seedset.data(), 
        seedpos.data(), labels);
    
    // convert local labels into global weights
    labels_to_weights<DType>(rank, size, n, avg_d, k, galpha, lalpha, labels,
        dataset, datapos, seedset, seedpos, weights);
    
    // release space
    delete[] labels;
    return seedpos.size()-1;
}

// -----------------------------------------------------------------------------
template<class DType>
int silk_seeding(                   // init k centers by silk
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   N,                            // total number of data points
    int   d,                            // real dimension of sparse data
    int   avg_d,                        // average dimension of sparse data
    int   k,                            // number of seeds
    int   m1,                           // #hash tables (1st level)
    int   h1,                           // #concat hash func (1st level)
    int   t,                            // threshold of #point IDs in a bucket
    int   m2,                           // #hash tables (2nd level)
    int   h2,                           // #concat hash func (2nd level)
    int   b,                            // threshold of #bucket IDs in a bin
    int   delta,                        // threshold of #point IDs in a bin
    float gbeta,                        // global beta
    float lbeta,                        // local  beta
    float galpha,                       // global alpha
    float lalpha,                       // local  alpha
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    std::vector<int> &seedset,          // seed set (return)
    std::vector<u64> &seedpos)          // seed position (return)
{
    // double start_wc_time = omp_get_wtime();
    
    // -------------------------------------------------------------------------
    //  silk overseeding
    // -------------------------------------------------------------------------
#ifdef DEBUG_INFO
    printf("Rank #%d: silk overseeding (n=%d)\n\n", rank, n);
#endif
    int K = 0;
    std::vector<int> over_seedset;
    std::vector<u64> over_seedpos;
    
    // get K seeds by silk
    K = silk_overseeding<DType>(rank, size, n, N, d, avg_d, m1, h1, t, m2, h2, 
        b, delta, gbeta, lbeta, galpha, lalpha, dataset, datapos, 
        over_seedset, over_seedpos);
    
    // stop early if have no larger than k seeds
    if (K <= k) {
        return early_stop(over_seedset, over_seedpos, seedset, seedpos);
    }
    // double over_time = omp_get_wtime() - start_wc_time;
    // printf("Rank #%d: K=%d, over_time=%.2lf\n\n", rank, K, over_time);
    
    // -------------------------------------------------------------------------
    //  get weights for K seeds by data assignment
    // -------------------------------------------------------------------------
#ifdef DEBUG_INFO
    printf("Rank #%d: data assignment => weights (n=%d, K=%d)\n\n", rank, n, K);
#endif
    int *weights = new int[K];
    
    // get weights for K seeds (remove seeds if they have no weights)
    K = get_weights_for_seeds<DType>(rank, size, n, avg_d, K, galpha, lalpha,
        dataset, datapos, over_seedset, over_seedpos, weights);
    
    // stop early if have no larger than k seeds
    if (K <= k) {
        delete[] weights;
        return early_stop(over_seedset, over_seedpos, seedset, seedpos);
    }
    // double weight_time = omp_get_wtime() - start_wc_time;
    // printf("Rank #%d: K=%d, weight_time=%.2lf\n\n", rank, K, weight_time-over_time);
    
    // -------------------------------------------------------------------------
    //  call k-means++ for final refinement using a single thread (@root)
    // -------------------------------------------------------------------------
#ifdef DEBUG_INFO
    printf("Rank #%d: k-means++ refinement (n=%d, k=%d)\n\n", rank, K, k);
#endif
    if (rank == 0) {
        // use OpenMP for k-means++ by default
        int *local_distinct_ids = new int[k];
        kmeanspp_seeding<int>(0, 1, K, K, avg_d, k, 0, over_seedset.data(),
            over_seedpos.data(), weights, local_distinct_ids, seedset, seedpos);
        delete[] local_distinct_ids;
    }
    // broadcast seedset and seedpos from root to other threads
    if (size > 1) broadcast_set_and_pos(rank, size, seedset, seedpos);
    
    // release space
    std::vector<int>().swap(over_seedset);
    std::vector<u64>().swap(over_seedpos);
    delete[] weights;
    
    // double kpp_time = omp_get_wtime() - start_wc_time;
    // printf("Rank #%d: k=%d, kpp_time=%.2lf\n\n", rank, k, kpp_time-weight_time);
    
    return k;
}

} // end namespace clustering
