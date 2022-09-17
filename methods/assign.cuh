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
template<class DType>
void global_frequent_items(         // apply frequent items for global data
    int   num,                          // number of point IDs in a bin
    float galpha,                       // global alpha
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    const int   *bin,                   // bin
    std::vector<int> &seedset,          // seed set (return)
    std::vector<u64> &seedpos)          // seed position (return)
{
    // get the total number of coordinates (total_num) in this bin
    u64 total_num = 0UL;
    for (int i = 0; i < num; ++i) {
        int id = bin[i]; total_num += datapos[id+1] - datapos[id];
    }
    
    // init an array to store all coordinates in this bin
    DType *arr = new DType[total_num];
    u64 cnt = 0UL;
    for (int i = 0; i < num; ++i) {
        const u64   *pos  = datapos + bin[i]; // get data pos
        const DType *data = dataset + pos[0]; // get data
        int len = get_length(0, pos);         // get data len
        
        // copy the coordinates of this data to the array
        std::copy(data, data+len, arr+cnt);
        cnt += len;
    }
    assert(cnt == total_num);
    
    // get the distinct coordinates and their frequencies
    DType *coord = new DType[total_num];
    int   *freq  = new int[total_num];
    int    n     = 0;
    int max_freq = distinct_coord_and_freq<DType>(total_num, arr, coord, freq, n);
    
    // get the high frequent coord and their freq as seedset and seedpos
    if (max_freq == 1) {
        // directly use all coords as seed and update seedset & seedpos
        seedpos.push_back(n);
        seedset.insert(seedset.end(), coord, coord+n);
    }
    else {
        // get the high frequent coordinates as seed 
        int threshold = (int) ceil((double) max_freq*galpha);
        DType *seed = new DType[n];
        cnt = 0; // number of coordinates for seed
        for (int i = 0; i < n; ++i) {
            if (freq[i] >= threshold) { seed[cnt++] = coord[i]; }
        }
        // update seedset and seedpos
        seedpos.push_back(cnt); // add cnt to seedpos
        seedset.insert(seedset.end(), seed, seed+cnt);
        delete[] seed;
    }
    // release space
    delete[] arr;
    delete[] coord;
    delete[] freq;
}

// -----------------------------------------------------------------------------
template<class DType>
void local_frequent_items(          // apply frequent items for local data
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   num,                          // number of point IDs in a bin
    int   local_seed_dim,               // local seed dimension
    float lalpha,                       // local alpha
    const DType *dataset,               // data set
    const u64 *datapos,                 // data position
    const int *bin,                     // bin
    int   *local_seed,                  // local seed (return)
    int   *local_freq,                  // local freq (return)
    int   &local_len)                   // actual length of local seed (return)
{
    // get the total number of coordinates (total_num) in this bin
    int lower_bound = n * rank;
    int upper_bound = n + lower_bound;
    u64 total_num = 0UL;
    for (int i = 0; i < num; ++i) {
        int id = bin[i]; // only consider local data
        if (id < lower_bound || id >= upper_bound) continue;
        
        // sum up the total_num
        id -= lower_bound;
        total_num += datapos[id+1] - datapos[id];
    }
    if (total_num == 0) return;
    
    // init an array to store all coordinates in this bin
    DType *arr = new DType[total_num];
    u64 cnt = 0UL;
    for (int i = 0; i < num; ++i) {
        int id = bin[i]; // only consider local data
        if (id < lower_bound || id >= upper_bound) continue;
        
        // copy the coordinates of this data to the array
        id -= lower_bound;                 // get data id
        int len = get_length(id, datapos); // get data len
        const DType *data = dataset + datapos[id]; // get data
        
        std::copy(data, data+len, arr+cnt);
        cnt += len;
    }
    assert(cnt == total_num);
    
    // get the distinct coordinates and their frequencies
    DType *coord = new DType[total_num];
    int   *freq  = new int[total_num];
    
    int coord_size = 0;
    int max_freq = distinct_coord_and_freq<DType>(total_num, arr, coord, 
        freq, coord_size);
    
    // get the high frequent coord and their freq as local_seed and local_freq
    int threshold = (int) ceil((double) max_freq*lalpha);
    if (max_freq == 1) {
        local_len = std::min(coord_size, local_seed_dim);
        std::copy(coord, coord + local_len, local_seed);
        std::copy(freq,  freq  + local_len, local_freq);
    }
    else {
        local_len = 0;
        for (int i = 0; i < coord_size; ++i) {
            if (freq[i] >= threshold) {
                local_seed[local_len] = (int) coord[i];
                local_freq[local_len] = freq[i];
                if (++local_len >= local_seed_dim) break;
            }
        }
    }
    // release space
    delete[] coord;
    delete[] freq;
    delete[] arr;
}

// -----------------------------------------------------------------------------
void frequent_items(                // apply frequent items for cand_list
    float galpha,                       // global alpha
    std::vector<std::pair<int,int> > &cand_list, // cand_list (allow modify)
    std::vector<int> &seedset,          // seed set (return)
    std::vector<u64> &seedpos);         // seed position (return)

// -----------------------------------------------------------------------------
template<class DType>
void bins_to_seeds(                 // convert global bins into global seeds
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   k,                            // number of bins (and seeds)
    int   avg_d,                        // average dimension of data points
    float galpha,                       // global alpha
    float lalpha,                       // local  alpha
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    const int   *binset,                // bin set
    const u64   *binpos,                // bin position
    std::vector<int> &seedset,          // seed set (return)
    std::vector<u64> &seedpos)          // seed position (return)
{
    // clear seedset and seedpos
    std::vector<int>().swap(seedset);
    std::vector<u64>().swap(seedpos);
    
    // estimate cost for seedset and seedpos
    srand(RANDOM_SEED);            // fix a random seed
    seedset.reserve((u64)k*avg_d); // estimation, not correct
    seedpos.reserve(k+1);          // correct reservation
    seedpos.push_back(0UL);
    
    // -------------------------------------------------------------------------
    //  calc k local seeds
    // -------------------------------------------------------------------------
    int local_seed_dim = avg_d*10000; // estimation, has protection
    int *local_seed = new int[local_seed_dim];
    int *local_freq = new int[local_seed_dim];
    
    int *all_buff = new int[MAX_BUFF_SIZE];
    int  max_buff = MAX_BUFF_SIZE / size;
    
    int  j = 0; // the counter of #buffers, start from 0
    std::vector<std::vector<int> > buffer;
    std::unordered_map<int, std::vector<std::pair<int,int> > > cand_list;
    
    for (int i = 0; i < k; ++i) {
        const int *bin = binset + binpos[i];  // get bin
        int bin_size = get_length(i, binpos); // get bin size
        
        if (size == 1) {
            // -----------------------------------------------------------------
            //  single-thread case: apply frequent items to get seeds
            // -----------------------------------------------------------------
            global_frequent_items<DType>(bin_size, galpha, dataset, datapos, 
                bin, seedset, seedpos);
        }
        else {
            // -----------------------------------------------------------------
            //  multi-thread case: apply frequent items to get local_seed and 
            //  local_freq & add them to buffer for further processing
            // -----------------------------------------------------------------
            if (i == 0) {
                buffer.push_back(std::vector<int>()); // init buffer
                buffer[j].reserve(max_buff);
            }
            int len = 0; // length for local seed (should <= local_seed_dim)
            local_frequent_items<DType>(rank, size, n, bin_size, local_seed_dim,
                lalpha, dataset, datapos, bin, local_seed, local_freq, len);
            if (len == 0) continue; // no dimension in this seed
            if (len*2+2 > max_buff) len = max_buff/2-1; // cut tails if too many
            
            // add local_seed & local_freq to buffer for further processing
            if (buffer[j].size()+len*2+2 > max_buff) {
                ++j;
                buffer.push_back(std::vector<int>());
                buffer[j].reserve(max_buff);
            }
            buffer[j].push_back(i);
            buffer[j].push_back(len);
            buffer[j].insert(buffer[j].end(), local_seed, local_seed+len);
            buffer[j].insert(buffer[j].end(), local_freq, local_freq+len);
        }
    }
    // -------------------------------------------------------------------------
    //  multi-thread case: calc k global seeds
    // -------------------------------------------------------------------------
    if (size > 1) {
        // get max_round based on j (local #buffers)
        int max_round = get_max_round(size, j); // start from 1
        
        for (int i = 0; i < max_round; ++i) {
            // gather all buffers from different local buffers to root
            if (j < i) buffer.push_back(std::vector<int>());
            int tlen = gather_all_buffers(size, buffer[i], all_buff);

            // @root: convert all buffers into candidate list
            if (rank == 0) all_buff_to_cand_list(tlen, all_buff, cand_list);
        }
        // @root: convert candidate list into global seeds
        if (rank == 0) {
            for (int i = 0; i < k; ++i) {
                frequent_items(galpha, cand_list[i], seedset, seedpos);
            }
        }
        // clear space
        std::vector<std::vector<int> >().swap(buffer);
        cand_list.clear();
    }
    delete[] local_seed;
    delete[] local_freq;
    delete[] all_buff;
    
    // accumulate the seed size to get the start position of each seed
    size_t n_seedpos = seedpos.size();
    for (size_t i = 1; i < n_seedpos; ++i) seedpos[i] += seedpos[i-1];

    // broadcast global seeds from root to other threads
    if (size > 1) broadcast_set_and_pos(rank, size, seedset, seedpos);
    assert(seedpos.size() == k+1);
    
#ifdef DEBUG_INFO
    printf("Rank #%d: avg_d=%d, total=%d\n", rank, seedpos[k]/k, seedpos[k]);
#endif
}

// -----------------------------------------------------------------------------
template<class DType>
void approx_assign_data(            // approximate sparse data assginment
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    const int   *seedset,               // seed set
    const u64   *seedpos,               // seed position
    int   *labels);                     // cluster labels for dataset (return)
    
// -----------------------------------------------------------------------------
template<class DType>
void exact_assign_data(             // exact sparse data assginment
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const DType *dataset,               // data set
    const u64   *datapos,               // data position
    const int   *seedset,               // seed set
    const u64   *seedpos,               // seed position
    int   *labels);                     // cluster labels for dataset (return)

} // end namespace clustering
