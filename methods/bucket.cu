#include "bucket.h"

namespace clustering {

// -----------------------------------------------------------------------------
int* shift_hash_results(            // shift e2lsh hash results
    int size,                           // number of MPIs (size)
    int n,                              // number of data points
    int m,                              // number of hash tables
    int *hash_results)                  // hash results (return)
{
    assert(m % size == 0); // distributed condition
    int bundle_num = m/size; assert((u64) n*size < MAX_INT);
    int bundle_len = n*size; 
    
    // redistribute the hash results before generating buckets
    int *new_results = new int[(u64) bundle_len*bundle_num];
    for (int i = 0; i < m; ++i) {
        int root = i / bundle_num;
        int j    = i % bundle_num;
        
        // the root will get the results and store them in rank order
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gather(hash_results + (u64) i*n, n, MPI_INT,  // send
            new_results + (u64) j*bundle_len, n, MPI_INT, // receive
            root, MPI_COMM_WORLD); // root: rank of the receiving process
        MPI_Barrier(MPI_COMM_WORLD);
    }
    delete[] hash_results;
    
    return new_results;
}

// -----------------------------------------------------------------------------
void extract_buckets(               // extract buckets from hash results
    int n,                              // number of data points
    int t,                              // threshold of #point IDs in a bucket
    int *index,                         // index (allow modify)
    int *results,                       // hash results (allow modify)
    std::vector<int> &bktset,           // bucket set (return)
    std::vector<u64> &bktpos)           // bucket position (return)
{
    // sorting by keys with cuda
    thrust::sequence(index, index+n); // init index with 0,1,2,3,...
    thrust::sort_by_key(results, results+n, index);
    
    // extract buckets from the sorted results
    int start = 0, cnt = 1;
    for (size_t i = 1; i < n; ++i) {
        if (results[i] == results[i-1]) ++cnt;
        else {
            // meet a new key, store elements for the last key if cnt > t
            if (cnt > t) {
                const int *id = index + start;
                bktpos.push_back(cnt);
                bktset.insert(bktset.end(), id, id+cnt);
            }
            start = i; cnt = 1;
        }
    }
    // deal with the last key
    if (cnt > t) {
        const int *id = index + start;
        bktpos.push_back(cnt);
        bktset.insert(bktset.end(), id, id+cnt);
    }
}

// -----------------------------------------------------------------------------
int hash_results_to_buckets(        // convert local hash results into buckets
    int rank,                           // MPI rank
    int size,                           // number of MPIs (size)
    int n,                              // number of data points
    int m,                              // signature size
    int t,                              // threshold of #point IDs in a bucket
    int *hash_results,                  // hash results (allow modify)
    std::vector<int> &bktset,           // bucket set (return)
    std::vector<u64> &bktpos)           // bucket position (return)
{
    // clear space for bktset and bktpos
    std::vector<int>().swap(bktset);
    std::vector<u64>().swap(bktpos);
    
    // recompute the number and length if multi-threads
    assert(m % size == 0);
    int bundle_num = m/size; assert((u64) n*size < MAX_INT);
    int bundle_len = n*size; 
    
    bktset.reserve((u64) bundle_num*bundle_len);     // over-estimation
    bktpos.reserve((u64) bundle_num*bundle_len + 1); // over-estimation
    bktpos.push_back(0UL);
    
    int *index = new int[bundle_len];
    for (int i = 0; i < bundle_num; ++i) {
#ifdef DEBUG_INFO
        printf("Rank #%d: bundle_num=%d, i=%d\n", rank, bundle_num, i+1);
#endif
        // extract buckets from hash results
        int *results = hash_results + (u64) i*bundle_len;
        extract_buckets(bundle_len, t, index, results, bktset, bktpos);
    }
    delete[] index;
    
    // accumulate the bucket size to get the start position of each bucket
    size_t n_bktpos = bktpos.size(); assert(n_bktpos <= MAX_INT);
    for (size_t i = 1; i < n_bktpos; ++i) bktpos[i] += bktpos[i-1];
    
    return n_bktpos-1;
}

// -----------------------------------------------------------------------------
int bins_to_buckets(                // convert global bins into local buckets
    int rank,                           // MPI rank
    int size,                           // number of MPIs (size)
    int n,                              // number of bins
    std::vector<int> &binset,           // bin set (allow modify)
    std::vector<u64> &binpos,           // bin position (allow modify)
    std::vector<int> &bktset,           // bucket set (return)
    std::vector<u64> &bktpos)           // bucket position (return)
{
    // clear bktset and bktpos
    std::vector<int>().swap(bktset);
    std::vector<u64>().swap(bktpos);
    
    if (size == 1) {
        // single-thread: directly swap the contents between bucket and bin
        bktset.swap(binset);
        bktpos.swap(binpos);
    }
    else {
        // multi-thread: calc start pos and the num of bins for this thread
        int num = n / size; if (n%size != 0) ++num;
        int start = rank * num;
        if (start + num > n) num = n - start;
        
        // copy binset to bktset for this thread
        const int *bin = binset.data() + binpos[start];
        u64 bin_size = binpos[start+num] - binpos[start];
        
        bktset.resize(bin_size);
        std::copy(bin, bin + bin_size, bktset.data());
        
        // compute and copy binpos to bktpos for this thread
        bktpos.resize(num+1);
        for (int i = 0; i <= num; ++i) {
            bktpos[i] = binpos[i+start] - binpos[start];
        }
        // clear binset and binpos
        std::vector<int>().swap(binset);
        std::vector<u64>().swap(binpos);
    }
    return bktpos.size() - 1;
}

} // end namespace clustering
