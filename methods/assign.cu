#include "assign.cuh"

namespace clustering {

// -----------------------------------------------------------------------------
struct PairCmp {
    __host__ __device__
    bool operator() (const std::pair<int,int> &o1, const std::pair<int,int> &o2)
    {
        return o1.first < o2.first;
    }
};

// -----------------------------------------------------------------------------
void frequent_items(                // apply frequent items for cand_list
    float galpha,                       // global alpha
    std::vector<std::pair<int,int> > &cand_list, // cand_list (allow modify)
    std::vector<int> &seedset,          // seed set (return)
    std::vector<u64> &seedpos)          // seed position (return)
{
    // first sort cand_list in ascending order by PairCmp
    thrust::sort(cand_list.begin(), cand_list.end(), PairCmp());
    
    // get the distinct coordinates and their frequencies
    u64 total_num = cand_list.size();
    int *coord = new int[total_num];
    int *freq  = new int[total_num];
    
    int cnt = cand_list[0].second; // get the frequency of coordinate
    int max_freq = -1, num = 0;
    for (size_t i = 1; i < total_num; ++i) {
        if (cand_list[i].first != cand_list[i-1].first) {
            coord[num] = cand_list[i-1].first; freq[num] = cnt;
            if (cnt > max_freq) max_freq = cnt;
            
            cnt = cand_list[i].second; ++num;
        }
        else cnt += cand_list[i].second;
    }
    coord[num] = cand_list[total_num-1].first; freq[num] = cnt;
    if (cnt > max_freq) max_freq = cnt;
    ++num;
    
    // get the high frequent coord and their freq as seedset and seedpos
    int threshold = (int) ceil((double) max_freq*galpha);
    if (max_freq == 1) {
        // directly use all coords as seed and update seedset & seedpos
        seedpos.push_back(num);
        seedset.insert(seedset.end(), coord, coord+num);
    }
    else {
        // get the high frequent coordinates as seed
        int *seed = new int[num];
        cnt = 0; // number of coordinates for seed
        for (int i = 0; i < num; ++i) {
            if (freq[i] >= threshold) { seed[cnt++] = coord[i]; }
        }
        // update seedset and seedpos
        seedpos.push_back(cnt); // add cnt to seedpos
        seedset.insert(seedset.end(), seed, seed+cnt);
        delete[] seed;
    }
    // release space
    std::vector<std::pair<int,int> >().swap(cand_list);
    delete[] coord;
    delete[] freq;
}

// -----------------------------------------------------------------------------
template<class DType>
__global__ void calc_jaccard_dist(  // calc jaccard distance
    int   batch,                        // batch number of data points
    int   k,                            // number of seeds
    const DType *d_dset,                // data set
    const u64   *d_dpos,                // data position
    const int   *d_sset,                // seed set
    const u64   *d_spos,                // seed position
    u16   *d_dist)                      // jaccard distance (return)
{
    u64 tid = (u64) blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (u64) batch * k) {
        const u64   *dpos = d_dpos + (u64) tid/k;
        const DType *data = d_dset + dpos[0];
        int n_data = get_length(0, dpos);
        
        const u64 *spos = d_spos + (u64) tid%k;
        const int *seed = d_sset + spos[0];
        int n_seed = get_length(0, spos);
        
        float jaccard = jaccard_dist<DType>(n_data, n_seed, data, seed);
        d_dist[tid] = (u16) ceil(jaccard * 65535);
    }
}

// -----------------------------------------------------------------------------
__global__ void nearest_seed(       // find the nearest seed for batch data
    int   batch,                        // batch number of data points
    int   k,                            // number of seeds
    const u16 *d_dist,                  // jaccard distance array
    int   *d_labels)                    // labels (return)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batch) {
        const u16 *dist = d_dist + (u64) tid*k; // get dist array
        
        // find the minimum jaccard distance for batch data
        int min_id   = 0;
        u16 min_dist = dist[0];
        for (int i = 1; i < k; ++i) {
            if (dist[i] < min_dist) { min_id = i; min_dist = dist[i]; }
        }
        d_labels[tid] = min_id;
    }
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
    int   *labels)                      // cluster labels for dataset (return)
{
    cudaSetDevice(DEVICE_LOCAL_RANK);

    // declare parameters and allocation
    u64 len = seedpos[k];
    int *d_sset; cudaMalloc((void**) &d_sset, sizeof(int)*len);
    u64 *d_spos; cudaMalloc((void**) &d_spos, sizeof(u64)*(k+1));
    
    cudaMemcpy(d_sset, seedset, sizeof(int)*len,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_spos, seedpos, sizeof(u64)*(k+1), cudaMemcpyHostToDevice);

    // mem_avail = total_mem - memory(d_sset + d_spos)
    u64 mem_avail = GPU_MEMORY_LIMIT - (sizeof(int)*len + sizeof(u64)*(k+1));
    u64 mem_usage = 0UL, n_dset = 0UL;
    int n_dpos = 0, batch = 0, start  = 0;
    
    for (int i = 0; i <= n; ++i) {
        // ---------------------------------------------------------------------
        //  calculate memory usage requirement if adding one more data
        // ---------------------------------------------------------------------
        if (i < n) {
            // d_dset + d_dpos + d_dist + d_labels
            n_dset = datapos[i+1] - datapos[start];
            n_dpos = batch + 2;
            mem_usage = sizeof(DType)*n_dset + sizeof(u64)*n_dpos + 
                (sizeof(u16)*k+sizeof(int))*(batch+1);
        }
        // ---------------------------------------------------------------------
        //  parallel batch data assignment if over mem_avail or end
        // ---------------------------------------------------------------------
        if (mem_usage > mem_avail || i == n) {
            n_dset = datapos[i] - datapos[start];
            n_dpos = batch + 1;
            mem_usage = sizeof(DType)*n_dset + sizeof(u64)*n_dpos + 
                (sizeof(u16)*k+sizeof(int))*batch;
#ifdef DEBUG_INFO
            printf("Rank #%d: n=%d, i=%d, batch=%d, mem_usage=%lu, mem_avail=%lu\n",
                rank, n, i, batch, mem_usage, mem_avail);
#endif
            // cuda allocation and memory copy from CPU to GPU
            const DType *h_dset = dataset + datapos[start];
            u64   *h_dpos = new u64[n_dpos];
            copy_pos(n_dpos, datapos + start, h_dpos);
            
            DType *d_dset;   cudaMalloc((void**) &d_dset,   sizeof(DType)*n_dset);
            u64   *d_dpos;   cudaMalloc((void**) &d_dpos,   sizeof(u64)*n_dpos);
            u16   *d_dist;   cudaMalloc((void**) &d_dist,   sizeof(u16)*k*batch);
            int   *d_labels; cudaMalloc((void**) &d_labels, sizeof(int)*batch);
            
            cudaMemcpy(d_dset, h_dset, sizeof(DType)*n_dset, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dpos, h_dpos, sizeof(u64)*n_dpos,   cudaMemcpyHostToDevice);
            
            // calc Jaccard distance between batch data and k seeds
            int block = BLOCK_SIZE;
            int grid  = ((u64) batch*k + block-1) / block;
            calc_jaccard_dist<DType><<<grid, block>>>(batch, k, d_dset, d_dpos,
                d_sset, d_spos, d_dist);
            
            // find the nearest seed for batch data
            grid = ((u64) batch + block-1) / block;
            nearest_seed<<<grid, block>>>(batch, k, d_dist, d_labels);
            
            // update labels & release local space
            cudaMemcpy(labels+start, d_labels, sizeof(int)*batch, cudaMemcpyDeviceToHost);
            
            cudaFree(d_dset); cudaFree(d_dpos);
            cudaFree(d_dist); cudaFree(d_labels);
            delete[] h_dpos;
            
            // update local parameters for next batch data assignment
            start += batch; batch = 0;
        }
        if (i < n) ++batch;
    }
    assert(start == n);
    cudaFree(d_sset); cudaFree(d_spos);
}

// -----------------------------------------------------------------------------
template void approx_assign_data(   // approximate sparse data assginment
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const u08 *dataset,                 // data set
    const u64 *datapos,                 // data position
    const int *seedset,                 // seed set
    const u64 *seedpos,                 // seed position
    int   *labels);                     // cluster labels for data (return)
    
// -----------------------------------------------------------------------------
template void approx_assign_data(   // approximate sparse data assginment
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const u16 *dataset,                 // data set
    const u64 *datapos,                 // data position
    const int *seedset,                 // seed set
    const u64 *seedpos,                 // seed position
    int   *labels);                     // cluster labels for data (return)
    
// -----------------------------------------------------------------------------
template void approx_assign_data(   // approximate sparse data assginment
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const int *dataset,                 // data set
    const u64 *datapos,                 // data position
    const int *seedset,                 // seed set
    const u64 *seedpos,                 // seed position
    int   *labels);                     // cluster labels for data (return)
    
// -----------------------------------------------------------------------------
template void approx_assign_data(   // approximate sparse data assginment
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const f32 *dataset,                 // data set
    const u64 *datapos,                 // data position
    const int *seedset,                 // seed set
    const u64 *seedpos,                 // seed position
    int   *labels);                     // cluster labels for data (return)

// -----------------------------------------------------------------------------
template<class DType>
__global__ void calc_jaccard_dist(  // calc jaccard distance
    int   batch,                        // batch number of data points
    int   k,                            // number of seeds
    const DType *d_dset,                // data set
    const u64   *d_dpos,                // data position
    const int   *d_sset,                // seed set
    const u64   *d_spos,                // seed position
    float *d_dist)                      // jaccard distance (return)
{
    u64 tid = (u64) blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (u64) batch * k) {
        const u64   *dpos = d_dpos + (u64) tid/k;
        const DType *data = d_dset + dpos[0];
        int n_data = get_length(0, dpos);
        
        const u64 *spos = d_spos + (u64) tid%k;
        const int *seed = d_sset + spos[0];
        int n_seed = get_length(0, spos);
        
        d_dist[tid] = jaccard_dist<DType>(n_data, n_seed, data, seed);
    }
}

// -----------------------------------------------------------------------------
__global__ void nearest_seed(       // find the nearest seed id for batch data
    int   batch,                        // batch number of data points
    int   k,                            // number of seeds
    const float *d_dist,                // jaccard distance array
    int   *d_labels)                    // labels (return)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batch) {
        const float *dist = d_dist + (u64) tid*k; // get dist array
        
        // find the minimum jaccard distance for batch data
        int   min_id   = 0;
        float min_dist = dist[0];
        for (int i = 1; i < k; ++i) {
            if (dist[i] < min_dist) { min_id = i; min_dist = dist[i]; }
        }
        d_labels[tid] = min_id;
    }
}

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
    int   *labels)                      // cluster labels for dataset (return)
{
    cudaSetDevice(DEVICE_LOCAL_RANK);

    // declare parameters and allocation
    u64 len = seedpos[k];
    int *d_sset = nullptr; cudaMalloc((void**)&d_sset, sizeof(int)*len);
    u64 *d_spos = nullptr; cudaMalloc((void**)&d_spos, sizeof(u64)*(k+1));
    
    cudaMemcpy(d_sset, seedset, sizeof(int)*len,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_spos, seedpos, sizeof(u64)*(k+1), cudaMemcpyHostToDevice);

    // mem_avail = total_mem - memory(d_sset + d_spos)
    u64 mem_avail = GPU_MEMORY_LIMIT - (sizeof(int)*len + sizeof(u64)*(k+1));
    u64 mem_usage = 0UL, n_dset = 0UL;
    int n_dpos = 0, batch = 0, start = 0;
    
    for (int i = 0; i <= n; ++i) {
        // ---------------------------------------------------------------------
        //  calculate memory usage requirement if adding one more data
        // ---------------------------------------------------------------------
        if (i < n) {
            // d_dset + d_dpos + d_dist + d_labels
            n_dset = datapos[i+1] - datapos[start];
            n_dpos = batch + 2;
            mem_usage = sizeof(DType)*n_dset + sizeof(u64)*n_dpos + 
                (sizeof(float)*k+sizeof(int))*(batch+1);
        }
        // ---------------------------------------------------------------------
        //  parallel batch data assignment if over mem_avail or end
        // ---------------------------------------------------------------------
        if (mem_usage > mem_avail || i == n) {
            n_dset = datapos[i] - datapos[start];
            n_dpos = batch + 1;
            mem_usage = sizeof(DType)*n_dset + sizeof(u64)*n_dpos + 
                (sizeof(float)*k+sizeof(int))*batch;
#ifdef DEBUG_INFO
            printf("Rank #%d: n=%d, i=%d, batch=%d, mem_usage=%lu, mem_avail=%lu\n",
                rank, n, i, batch, mem_usage, mem_avail);
#endif
            // cuda allocation and memory copy from CPU to GPU
            const DType *h_dset = dataset + datapos[start];
            u64   *h_dpos = new u64[n_dpos];
            copy_pos(n_dpos, datapos + start, h_dpos);
            
            DType *d_dset;   cudaMalloc((void**) &d_dset,   sizeof(DType)*n_dset);
            u64   *d_dpos;   cudaMalloc((void**) &d_dpos,   sizeof(u64)*n_dpos);
            float *d_dist;   cudaMalloc((void**) &d_dist,   sizeof(float)*batch*k);
            int   *d_labels; cudaMalloc((void**) &d_labels, sizeof(int)*batch);
            
            cudaMemcpy(d_dset, h_dset, sizeof(DType)*n_dset, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dpos, h_dpos, sizeof(u64)*n_dpos,   cudaMemcpyHostToDevice);
            
            // compute Jaccard distance for between batch data and k seeds
            int block = BLOCK_SIZE;
            int grid  = ((u64) batch*k + block-1) / block;
            calc_jaccard_dist<DType><<<grid, block>>>(batch, k, d_dset, d_dpos,
                d_sset, d_spos, d_dist);
            
            // find the nearest seed for batch data
            grid = ((u64) batch + block-1) / block;
            nearest_seed<<<grid, block>>>(batch, k, d_dist, d_labels);
            
            // update labels & release local space
            cudaMemcpy(&labels[start], d_labels, sizeof(int)*batch, cudaMemcpyDeviceToHost);
            
            cudaFree(d_dset); cudaFree(d_dpos); 
            cudaFree(d_dist); cudaFree(d_labels);
            delete[] h_dpos;
            
            // update local parameters for next batch data assignment
            start += batch; batch = 0;
        }
        if (i < n) ++batch;
    }
    assert(start == n);
    cudaFree(d_sset); cudaFree(d_spos);
}

// -----------------------------------------------------------------------------
template void exact_assign_data(    // exact sparse data assginment
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const u08 *dataset,                 // data set
    const u64 *datapos,                 // data position
    const int *seedset,                 // seed set
    const u64 *seedpos,                 // seed position
    int   *labels);                     // cluster labels for data (return)
    
// -----------------------------------------------------------------------------
template void exact_assign_data(    // exact sparse data assginment
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const u16 *dataset,                 // data set
    const u64 *datapos,                 // data position
    const int *seedset,                 // seed set
    const u64 *seedpos,                 // seed position
    int   *labels);                     // cluster labels for data (return)
    
// -----------------------------------------------------------------------------
template void exact_assign_data(    // exact sparse data assginment
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const int *dataset,                 // data set
    const u64 *datapos,                 // data position
    const int *seedset,                 // seed set
    const u64 *seedpos,                 // seed position
    int   *labels);                     // cluster labels for data (return)
    
// -----------------------------------------------------------------------------
template void exact_assign_data(    // exact sparse data assginment
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const f32 *dataset,                 // data set
    const u64 *datapos,                 // data position
    const int *seedset,                 // seed set
    const u64 *seedpos,                 // seed position
    int   *labels);                     // cluster labels for data (return)

} // end namespace clustering
