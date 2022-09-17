#include "seeding.cuh"

namespace clustering {

// -----------------------------------------------------------------------------
void generate_k_distinct_ids(       // generate k distinct ids
    int k,                              // k value
    int n,                              // total range
    int *distinct_ids)                  // distinct ids (return)
{
    bool *select = new bool[n]; memset(select, false, sizeof(bool)*n);
    int  id = -1;
    for (int i = 0; i < k; ++i) {
        // every time draw a distinct id uniformly at from from [0,n-1]
        do { id = uniform_u32(0, n-1); } while (select[id]);
        
        select[id] = true;
        distinct_ids[i] = id;
    }
    delete[] select;
}

// -----------------------------------------------------------------------------
void gather_all_local_seedset(      // gather all local seedsets to root
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    const std::vector<int> &local_seedset, // local seedset
    std::vector<int> &seedset)          // seedset at root (return)
{
    // gather the length of local_seedset from different threads to root
    int len = (int) local_seedset.size(); // length of local_seedset
    int *rlen = new int[size];
    for (int i = 0; i < size; ++i) rlen[i] = 0;
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(&len, 1, MPI_INT, rlen, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // @root: get the total length of global seedset & resize it
    int tlen = 0;
    for (int i = 0; i < size; ++i) tlen += rlen[i];
    if (rank == 0) { assert(tlen <= MAX_BUFF_SIZE); seedset.resize(tlen); }
    
    // @root: init displacements to gather all local_seedset
    int *displs = new int[size];
    displs[0] = 0;
    for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + rlen[i-1];

    // gather local_seedset from different threads to global seedset @root
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(local_seedset.data(), len, MPI_INT, seedset.data(), rlen, 
        displs, MPI_INT, 0, MPI_COMM_WORLD); // 0 is root location
    MPI_Barrier(MPI_COMM_WORLD);
    
    delete[] rlen;
    delete[] displs;
}

// -----------------------------------------------------------------------------
void gather_all_local_seedpos(      // gather all local seedpos to root
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   k,                            // k value
    const std::vector<u64> &local_seedpos, // local seedpos
    std::vector<u64> &seedpos)          // seedpos at root (return)
{
    // gather the length of local_seedpos from different threads to root
    int len = (int) local_seedpos.size()-1; // skip the first 0
    int *rlen = new int[size];
    for (int i = 0; i < size; ++i) rlen[i] = 0;
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(&len, 1, MPI_INT, rlen, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // @root: get the total length of global seedpos & reinit it
    int tlen = 0;
    for (int i = 0; i < size; ++i) tlen += rlen[i];
    if (rank == 0) { assert(tlen == k); seedpos.resize(k+1); seedpos[0]=0; }
    
    // @root: init displacements to gather all local_seedpos
    int *displs = new int[size];
    displs[0] = 0;
    for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + rlen[i-1];

    // gather local_seedpos from different threads to global seedpos @root
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(local_seedpos.data()+1, len, MPI_UINT64_T, seedpos.data()+1, 
        rlen, displs, MPI_UINT64_T, 0, MPI_COMM_WORLD); // 0 is root location
    MPI_Barrier(MPI_COMM_WORLD);
    
    delete[] rlen;
    delete[] displs;
}

// -----------------------------------------------------------------------------
template<class DType>
__global__ void calc_jaccard_dist(  // calc jaccard distance
    int   batch,                        // batch number of data points
    int   n_seed,                       // length of seed
    const int   *d_seed,                // seed
    const DType *d_dset,                // data set
    const u64   *d_dpos,                // data position
    float *d_dist)                      // jaccard distance (return)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batch) {
        const u64   *dpos = &d_dpos[tid];
        const DType *data = &d_dset[dpos[0]];
        int n_data = get_length(0, dpos);
        
        float dist = jaccard_dist<DType>(n_data, n_seed, data, d_seed);
        if (d_dist[tid] > dist) d_dist[tid] = dist;
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
    float *nn_dist)                     // nn_dist (return)
{
    cudaSetDevice(DEVICE_LOCAL_RANK);
    
    // declare parameters and allocation
    int n_seed  = (int) seed.size();
    int *d_seed = nullptr; cudaMalloc((void**)&d_seed, sizeof(int)*n_seed);
    cudaMemcpy(d_seed, seed.data(), sizeof(int)*n_seed, cudaMemcpyHostToDevice);
    
    // mem_avail = total_mem - memory(d_seed)
    u64 mem_avail = GPU_MEMORY_LIMIT - sizeof(int)*n_seed;
    u64 mem_usage = 0UL, n_dset = 0UL;
    int n_dpos = 0, batch = 0, start  = 0;
    
    for (int i = 0; i <= n; ++i) {
        // ---------------------------------------------------------------------
        //  calculate memory usage requirement if adding one more data
        // ---------------------------------------------------------------------
        if (i < n) {
            // d_dset + d_dpos + d_dist
            n_dset = datapos[i+1] - datapos[start];
            n_dpos = batch + 2;
            mem_usage = sizeof(DType)*n_dset + sizeof(u64)*n_dpos + 
                sizeof(float)*(batch+1);
        }
        // ---------------------------------------------------------------------
        //  parallel nn_dist update for batch data if over mem_avail or end
        // ---------------------------------------------------------------------
        if (mem_usage > mem_avail || i == n) {
            n_dset = datapos[i] - datapos[start];
            n_dpos = batch + 1;
            mem_usage = sizeof(DType)*n_dset + sizeof(u64)*n_dpos + 
                sizeof(float)*batch;
#ifdef DEBUG_INFO
            printf("Rank #%d: n=%d, i=%d, batch=%d, mem_usage=%lu, mem_avail=%lu\n",
                rank, n, i, batch, mem_usage, mem_avail);
#endif
            // cuda allocation and memory copy from CPU to GPU 
            const DType *h_dset = dataset + datapos[start];
            float *h_dist = nn_dist + start; // allow modify
            u64   *h_dpos = new u64[n_dpos];
            copy_pos(n_dpos, datapos + start, h_dpos);
            
            DType *d_dset; cudaMalloc((void**) &d_dset, sizeof(DType)*n_dset);
            u64   *d_dpos; cudaMalloc((void**) &d_dpos, sizeof(u64)*n_dpos);
            float *d_dist; cudaMalloc((void**) &d_dist, sizeof(float)*batch);
            
            cudaMemcpy(d_dset, h_dset, sizeof(DType)*n_dset, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dpos, h_dpos, sizeof(u64)*n_dpos,   cudaMemcpyHostToDevice);
            cudaMemcpy(d_dist, h_dist, sizeof(float)*batch,  cudaMemcpyHostToDevice);
            
            // calc Jaccard distance for between batch data and seed
            int block = BLOCK_SIZE;
            int grid  = ((u64) batch + block-1) / block;
            calc_jaccard_dist<DType><<<grid, block>>>(batch, n_seed, d_seed, 
                d_dset, d_dpos, d_dist);
            
            // update nn_dist for batch data & release local space
            cudaMemcpy(h_dist, d_dist, sizeof(float)*batch, cudaMemcpyDeviceToHost);
            
            cudaFree(d_dset); cudaFree(d_dpos); cudaFree(d_dist);
            delete[] h_dpos;
            
            // update local parameters for next nn_dist update
            start += batch; batch = 0;
        }
        if (i < n) ++batch;
    }
    assert(start == n);
    cudaFree(d_seed);
}

// -----------------------------------------------------------------------------
template void update_nn_dist(       // update nn_dist for local data
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    const std::vector<int> &seed,       // last seed
    const u08 *dataset,                 // data set
    const u64 *datapos,                 // data position
    float *nn_dist);                    // nn_dist (return)

// -----------------------------------------------------------------------------
template void update_nn_dist(       // update nn_dist for local data
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    const std::vector<int> &seed,       // last seed
    const u16 *dataset,                 // data set
    const u64 *datapos,                 // data position
    float *nn_dist);                    // nn_dist (return)

// -----------------------------------------------------------------------------
template void update_nn_dist(       // update nn_dist for local data
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    const std::vector<int> &seed,       // last seed
    const int *dataset,                 // data set
    const u64 *datapos,                 // data position
    float *nn_dist);                    // nn_dist (return)

// -----------------------------------------------------------------------------
template void update_nn_dist(       // update nn_dist for local data
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    const std::vector<int> &seed,       // last seed
    const f32 *dataset,                 // data set
    const u64 *datapos,                 // data position
    float *nn_dist);                    // nn_dist (return)

// -----------------------------------------------------------------------------
void broadcast_target_data(         // broadcast target data to all threads
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    std::vector<int> &target_data)      // target data (return)
{
    // gather the length of target_data from different threads to root (rank=0)
    int len = (int) target_data.size(); // length of target_data
    int *rlen = new int[size];
    for (int i = 0; i < size; ++i) rlen[i] = 0;
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(&len, 1, MPI_INT, rlen, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // @root (rank=0): determine the rank of local_data as new_root
    int new_root = 0;
    for (int i = 0; i < size; ++i) if (rlen[i] > 0) new_root = i;
    
    // broadcast the new_root from old root (rank=0) to all threads
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&new_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // broadcast the len and the target_data from new_root to all threads
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&len, 1, MPI_INT, new_root, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank != new_root) target_data.resize(len);
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(target_data.data(), len, MPI_INT, new_root, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    delete[] rlen;
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
__global__ void update_nn_dist(     // update nn_dist for batch data
    int   batch,                        // batch number of data points
    int   k,                            // number of seeds
    const float *d_dist,                // jaccard distance array
    float *d_nn_dist)                   // nn_dist (return)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batch) {
        const float *dist = d_dist + (u64) tid*k; // get dist array
        float min_dist = d_nn_dist[tid];
        
        // update min_dist among the k dist
        for (int i = 0; i < k; ++i) {
            if (dist[i] < min_dist) min_dist = dist[i];
        }
        d_nn_dist[tid] = min_dist;
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
    float *nn_dist)                     // nn_dist (return)
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
    int n_dpos = 0, batch = 0, start  = 0;
    
    for (int i = 0; i <= n; ++i) {
        // assert(batch == i-start);
        // ---------------------------------------------------------------------
        //  calculate memory usage requirement if adding one more data
        // ---------------------------------------------------------------------
        if (i < n) {
            // d_dset + d_dpos + d_dist + d_nn_dist
            n_dset = datapos[i+1] - datapos[start];
            n_dpos = batch + 2;
            mem_usage = sizeof(DType)*n_dset + sizeof(u64)*n_dpos + 
                sizeof(float)*(k+1)*(batch+1);
        }
        // ---------------------------------------------------------------------
        //  parallel batch data assignment if over mem_avail or end
        // ---------------------------------------------------------------------
        if (mem_usage > mem_avail || i == n) {
            n_dset = datapos[i] - datapos[start];
            n_dpos = batch + 1;
            mem_usage = sizeof(DType)*n_dset + sizeof(u64)*n_dpos + 
                sizeof(float)*(k+1)*batch;
#ifdef DEBUG_INFO
            printf("Rank #%d: n=%d, i=%d, batch=%d, mem_usage=%lu, mem_avail=%lu\n",
                rank, n, i, batch, mem_usage, mem_avail);
#endif
            // cuda allocation and memory copy from CPU to GPU
            const DType *h_dset = dataset + datapos[start];
            float *h_nn_dist = nn_dist + start; // allow modify
            u64   *h_dpos = new u64[n_dpos];
            copy_pos(n_dpos, datapos + start, h_dpos);
            
            DType *d_dset;    cudaMalloc((void**) &d_dset,    sizeof(DType)*n_dset);
            u64   *d_dpos;    cudaMalloc((void**) &d_dpos,    sizeof(u64)*n_dpos);
            float *d_dist;    cudaMalloc((void**) &d_dist,    sizeof(float)*batch*k);
            float *d_nn_dist; cudaMalloc((void**) &d_nn_dist, sizeof(float)*batch);
            
            cudaMemcpy(d_dset,    h_dset,    sizeof(DType)*n_dset, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dpos,    h_dpos,    sizeof(u64)*n_dpos,   cudaMemcpyHostToDevice);
            cudaMemcpy(d_nn_dist, h_nn_dist, sizeof(float)*batch,  cudaMemcpyHostToDevice);
            
            // calc Jaccard distance between batch data and k seeds
            int block = BLOCK_SIZE;
            int grid  = ((u64) batch*k + block-1) / block;
            calc_jaccard_dist<DType><<<grid, block>>>(batch, k, d_dset, d_dpos,
                d_sset, d_spos, d_dist);
            
            // update the nn_dist for batch data
            grid = ((u64) batch + block-1) / block;
            update_nn_dist<<<grid, block>>>(batch, k, d_dist, d_nn_dist);
            
            // get the new nn_dist & release local space
            cudaMemcpy(h_nn_dist, d_nn_dist, sizeof(float)*batch, cudaMemcpyDeviceToHost);
            
            cudaFree(d_dset); cudaFree(d_dpos);
            cudaFree(d_dist); cudaFree(d_nn_dist);
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
template void update_nn_dist(       // update nn_dist for local data by k seeds
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const u08 *dataset,                 // data set
    const u64 *datapos,                 // data position
    const int *seedset,                 // seed set
    const u64 *seedpos,                 // seed position
    float *nn_dist);                    // nn_dist (return)
    
// -----------------------------------------------------------------------------
template void update_nn_dist(       // update nn_dist for local data by k seeds
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const u16 *dataset,                 // data set
    const u64 *datapos,                 // data position
    const int *seedset,                 // seed set
    const u64 *seedpos,                 // seed position
    float *nn_dist);                    // nn_dist (return)
    
// -----------------------------------------------------------------------------
template void update_nn_dist(       // update nn_dist for local data by k seeds
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const int *dataset,                 // data set
    const u64 *datapos,                 // data position
    const int *seedset,                 // seed set
    const u64 *seedpos,                 // seed position
    float *nn_dist);                    // nn_dist (return)
    
// -----------------------------------------------------------------------------
template void update_nn_dist(       // update nn_dist for local data by k seeds
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const f32 *dataset,                 // data set
    const u64 *datapos,                 // data position
    const int *seedset,                 // seed set
    const u64 *seedpos,                 // seed position
    float *nn_dist);                    // nn_dist (return)

// -----------------------------------------------------------------------------
void labels_to_weights(             // convert local labels to global weights
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const int *labels,                  // labels for n local data
    int   *weights)                     // weights for k seeds (return)
{
    assert((u64) n*size < MAX_INT); // total num of data points
    int N = n*size; 
    int *all_labels = new int[N];
    
    // -------------------------------------------------------------------------
    //  get all labels from different threads to root
    // -------------------------------------------------------------------------
    if (size == 1) {
        // directly copy labels to all labels
        std::copy(labels, labels + n, all_labels);
    }
    else {
        // get all labels from different threads to root
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gather(labels, n, MPI_INT, all_labels, n, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // -------------------------------------------------------------------------
    //  @root: sequentical counting the number of labels as weight for each seed
    // -------------------------------------------------------------------------
    if (rank == 0) {
        memset(weights, 0, sizeof(int)*k); // init weights
        int pos = -1;
        for (int i = 0; i < N; ++i) { pos = all_labels[i]; ++weights[pos]; }
    }
    delete[] all_labels;
}

// -----------------------------------------------------------------------------
int early_stop(                     // early stop process
    std::vector<int> &over_seedset,     // over seed set (allow modify)
    std::vector<u64> &over_seedpos,     // over seed position (allow modify)
    std::vector<int> &seedset,          // seed set (return)
    std::vector<u64> &seedpos)          // seed position (return)
{
    // clear original space for seedset & seedpos
    std::vector<int>().swap(seedset);
    std::vector<u64>().swap(seedpos);
    
    // swap contents for over_seedset and seedset
    seedset.swap(over_seedset);
    seedpos.swap(over_seedpos);
    
    return seedpos.size()-1;
}

} // end namespace clustering
