#include "bin.h"

namespace clustering {

// -----------------------------------------------------------------------------
int signature_to_index(             // convert signature into a set of bins
    int n,                              // length of signature
    int *signature,                     // signature (allow modify)
    int *index,                         // index of signature (return)
    int *index_key,                     // index key (return)
    int *index_pos)                     // index position (return)
{
    // sort signature and its corresponding index
    thrust::sequence(index, index+n); // init index with 0,1,2,3,...
    thrust::sort_by_key(signature, signature+n, index);
    
    // update index_pos
    int j = 0; index_pos[0] = 0;
    for (int i = 1; i < n; ++i) {
        if (signature[i] != signature[i-1]) {
            index_key[j++] = signature[i-1]; 
            index_pos[j] = i;
        }
    }
    index_key[j++] = signature[n-1];
    index_pos[j] = n;
    return j;
}

// -----------------------------------------------------------------------------
void get_all_point_ids(             // get all point IDs from bin
    int   num,                          // number of bucket IDs in a bin
    const int *bin,                     // bin
    const int *bktset,                  // bucket set
    const u64 *bktpos,                  // bucket position
    int   *tmp_arr,                     // tmp array (allow modify)
    u64   &total_num)                   // total num (return)
{
    // get total_num of point IDs for the buckets in this bin
    total_num = 0UL;
    for (int i = 0; i < num; ++i) {
        int id = bin[i]; total_num += bktpos[id+1] - bktpos[id];
    }
    if (total_num > STATIC_ARRAY_SIZE) return;
    
    // init an array to store all point IDs in this bin
    u64 cnt = 0UL;
    for (int i = 0; i < num; ++i) {
        const u64 *pos = bktpos + bin[i];
        const int *bkt = &bktset[pos[0]];  // get bucket
        int len = (int) (pos[1] - pos[0]); // get bucket len
        
        // copy this bucket (point IDs) to the array
        std::copy(bkt, bkt+len, tmp_arr+cnt);
        cnt += len;
    }
    assert(cnt == total_num);
}

// -----------------------------------------------------------------------------
void get_cand_bin(                  // get cand bin
    int   n,                            // number of distinct pIDs
    int   max_freq,                     // maximum frequency
    int   cand_bin_size,                // total cand_bin size
    float bin_freq,                     // bin freq to get frequent data
    const int *pid,                     // distinct point IDs
    const int *freq,                    // freq of distinct point IDs
    int   *cand_bin,                    // cand_bin (return)
    int   &cand_len)                    // actual length of cand_bin (return)
{
    assert(n <= cand_bin_size);
    int threshold = (int) ceil((double) max_freq*bin_freq);
    
    // get cand_bin & update cand_len
    if (max_freq == 1) {
        cand_len = n;
        std::copy(pid, pid + n, cand_bin);
    } 
    else {
        cand_len = 0;
        for (int i = 0; i < n; ++i) {
            if (freq[i] >= threshold) cand_bin[cand_len++] = pid[i];
        }
    }
    
    // drop this cand_bin if too large
    if (cand_len > CAND_LEN_SIZE) {
        // printf("cand_len=%d, max_freq=%d, thresold=%d\n", cand_len, max_freq, 
        //     threshold);
        cand_len = 0;
    }
}

// -----------------------------------------------------------------------------
int frequent_data(                  // get frequent data from buckets
    int   num,                          // number of bucket IDs in a bin
    int   cand_bin_size,                // total cand_bin size
    float bin_freq,                     // bin freq to get frequent data
    const int *bin,                     // bin
    const int *bktset,                  // bucket set
    const u64 *bktpos,                  // bucket position
    int   *tmp_arr,                     // tmp array (allow modify)
    int   *tmp_pid,                     // tmp pid list (allow modify)
    int   *tmp_freq,                    // tmp freq list (allow modify)
    int   *cand_bin,                    // cand bin (return)
    int   &cand_len)                    // actual length of cand_bin (return)
{
    // get all point IDs from the input bin
    u64 total_num = 0UL;
    get_all_point_ids(num, bin, bktset, bktpos, tmp_arr, total_num);
    
    if (total_num > STATIC_ARRAY_SIZE) {
        // printf("total_num = %lu\n", total_num);
        cand_len = 0;
        return 0;
    }
    
    // get the distinct point IDs and their frequencies
    int n = 0;
    int max_freq = distinct_coord_and_freq<int>(total_num, tmp_arr, tmp_pid, 
        tmp_freq, n);
    
    // get the high frequent point IDs & add them to cand_bin
    get_cand_bin(n, max_freq, cand_bin_size, bin_freq, tmp_pid, tmp_freq, 
        cand_bin, cand_len);
    
    return max_freq;
}

// -----------------------------------------------------------------------------
void add_cand_bin(                  // add cand_bin to bins
    int  len,                           // length of cand_bin
    int  delta,                         // delta
    bool filter,                        // whether filter the points in bin
    int  *cand_bin,                     // cand_bin (allow modify)
    std::vector<int> &binset,           // bin set (return)
    std::vector<u64> &binpos)           // bin position (return)
{
    if (len > delta) {
        // only keep at most MAX_BIN_SIZE point IDs if filter is true
        if (filter && len > MAX_BIN_SIZE) {
            std::random_shuffle(cand_bin, cand_bin+len);
            binpos.push_back(MAX_BIN_SIZE);
            binset.insert(binset.end(), cand_bin, cand_bin+MAX_BIN_SIZE);
        } else {
            binpos.push_back(len);
            binset.insert(binset.end(), cand_bin, cand_bin+len);
        }
    }
}

// -----------------------------------------------------------------------------
void index_to_bins(                 // convert index into bins
    int   num_keys,                     // number of keys
    int   cand_bin_size,                // total cand_bin size
    int   b,                            // threshold of #buckets in a bin
    int   delta,                        // threshold of #points  in a bin
    bool  filter,                       // whether filter the points in bin
    float gbeta,                        // global beta
    const int *bktset,                  // bucket set
    const u64 *bktpos,                  // bucket position
    const int *index,                   // index 
    const int *index_pos,               // index position
    int   *tmp_arr,                     // tmp array (allow modify)
    int   *tmp_pid,                     // tmp pid list (allow modify)
    int   *tmp_freq,                    // tmp freq list (allow modify)
    int   *cand_bin,                    // candidate bin (allow modify)
    std::vector<int> &binset,           // bin set (return)
    std::vector<u64> &binpos)           // bin position (return)
{
    // select bins and add to binset
    int len = 0; // actual length of cand_bin
    for (int i = 0; i < num_keys; ++i) {
        const int *bin = index + index_pos[i]; // get bin
        int num = index_pos[i+1]-index_pos[i]; // get bin num
        if (num <= b) continue; // skip if |bin| <= b
        
        // using global bin freq to get cand_bin
        frequent_data(num, cand_bin_size, gbeta, bin, bktset, bktpos,
            tmp_arr, tmp_pid, tmp_freq, cand_bin, len);
        
        // add cand_bin to bins
        add_cand_bin(len, delta, filter, cand_bin, binset, binpos);
    }
}

// -----------------------------------------------------------------------------
int index_to_buffer(                // convert index into buffer
    int   num_keys,                     // number of keys
    int   cand_bin_size,                // total cand_bin size
    int   max_buff,                     // maximum buffer size
    float lbeta,                        // local beta
    const int *bktset,                  // bucket set
    const u64 *bktpos,                  // bucket position
    const int *index,                   // index 
    const int *index_key,               // index key
    const int *index_pos,               // index position
    int   *tmp_arr,                     // tmp array (allow modify)
    int   *tmp_pid,                     // tmp pid list (allow modify)
    int   *tmp_freq,                    // tmp freq list (allow modify)
    int   *cand_bin,                    // candidate bin (allow modify)
    std::vector<std::vector<int> > &buffer) // buffer (return)
{
    int j = 0; // the counter of #buffers
    buffer.push_back(std::vector<int>());
    buffer[j].reserve(max_buff);
    
    for (int i = 0; i < num_keys; ++i) {
        const int *bin = index + index_pos[i]; // get bin
        int num = index_pos[i+1]-index_pos[i]; // get bin num
        int key = index_key[i];                // get bin key
        
        // using local bin freq to get cand_bin
        int len = 0; // actual length of cand_bin
        int max_freq = frequent_data(num, cand_bin_size, lbeta, bin, bktset, 
            bktpos, tmp_arr, tmp_pid, tmp_freq, cand_bin, len);
        
        if (len == 0) continue; // no data in this bin
        if (len+4 > max_buff) continue;
        
        // add cand_bin to buffer for further processing
        if (buffer[j].size()+len+4 > max_buff) {
            ++j;
            buffer.push_back(std::vector<int>());
            buffer[j].reserve(max_buff);
        }
        buffer[j].push_back(key);
        buffer[j].push_back(num);
        buffer[j].push_back(max_freq);
        buffer[j].push_back(len);
        buffer[j].insert(buffer[j].end(), cand_bin, cand_bin+len);
    }
    return j; // start from 0
}

// -----------------------------------------------------------------------------
struct PairCmp {
__host__ __device__
bool operator() (const std::pair<int,int> &o1, const std::pair<int,int> &o2) {
    return o1.first < o2.first;
}
};

// -----------------------------------------------------------------------------
void frequent_data(                 // get frequent data from cand_list
    int   cand_bin_size,                // total cand_bin size
    float gbeta,                        // global beta
    std::vector<std::pair<int,int> > &cand_list, // cand_list (allow modify)
    int   *tmp_pid,                     // tmp pid list (allow modify)
    int   *tmp_freq,                    // tmp freq list (allow modify)
    int   *cand_bin,                    // cand_bin (return)
    int   &cand_len)                    // actual length of cand_bin (return)
{
    // first sort cand_list in ascending order by PairCmp
    thrust::sort(cand_list.begin(), cand_list.end(), PairCmp());
    
    // get the distinct point IDs and their frequencies
    u64 total_num = cand_list.size();
    if (total_num > STATIC_ARRAY_SIZE) { 
        // printf("total_num = %lu\n", total_num);
        cand_len = 0;
        std::vector<std::pair<int,int> >().swap(cand_list);
        return;
    }
    
    int cnt = cand_list[0].second; // get the frequency of point ID
    int max_freq = -1, num = 0;
    for (size_t i = 1; i < total_num; ++i) {
        if (cand_list[i].first != cand_list[i-1].first) {
            tmp_pid[num] = cand_list[i-1].first; tmp_freq[num] = cnt;
            if (cnt > max_freq) max_freq = cnt;
            
            cnt = cand_list[i].second; ++num;
        }
        else cnt += cand_list[i].second;
    }
    tmp_pid[num] = cand_list[total_num-1].first; tmp_freq[num] = cnt;
    if (cnt > max_freq) max_freq = cnt;
    ++num;
    
    // get the high frequent point IDs & add them to cand_bin
    get_cand_bin(num, max_freq, cand_bin_size, gbeta, tmp_pid, tmp_freq, 
        cand_bin, cand_len);
    
    // release space
    std::vector<std::pair<int,int> >().swap(cand_list);
}

// -----------------------------------------------------------------------------
int signatures_to_bins(             // convert local signatures into global bins
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   N,                            // total number of data points
    int   n,                            // num of buckets / length of signatures
    int   l,                            // number of minhash tables
    int   b,                            // threshold of #bucket IDs in a bin
    int   delta,                        // threshold of #point  IDs in a bin
    bool  filter,                       // whether filter #point IDs in a bin
    float gbeta,                        // global beta
    float lbeta,                        // local  beta
    const int *bktset,                  // bucket set
    const u64 *bktpos,                  // bucket position
    int   *signatures,                  // minhash signatures (allow modify)
    std::vector<int> &binset,           // bin set (return)
    std::vector<u64> &binpos)           // bin position (return)
{
    // clear binset and binpos
    std::vector<int>().swap(binset);
    std::vector<u64>().swap(binpos);
    
    // estimate cost for binset and binpos
    srand(RANDOM_SEED);         // fix a random seed
    binset.reserve((u64) n*size*MAX_BIN_SIZE); // maybe under-estimate
    binpos.reserve((u64) n*size+1);  // worst case size
    binpos.push_back(0UL);
    
    // -------------------------------------------------------------------------
    //  convert local signatures into global binset and binpos
    // -------------------------------------------------------------------------
    int *tmp_arr   = new int[STATIC_ARRAY_SIZE];
    int *tmp_pid   = new int[STATIC_ARRAY_SIZE];
    int *tmp_freq  = new int[STATIC_ARRAY_SIZE];
    int *cand_bin  = new int[N];
    
    int *index     = new int[n];
    int *index_key = new int[n];   // worst case size
    int *index_pos = new int[n+1]; // worst case size
    
    int *all_buff  = new int[MAX_BUFF_SIZE];
    int  max_buff  = MAX_BUFF_SIZE / size;
    
    for (int i = 0; i < l; ++i) {
        // ---------------------------------------------------------------------
        //  local signature to local bucket index, index_key, and index_pos
        // ---------------------------------------------------------------------
        int num_keys = signature_to_index(n, signatures + (u64) i*n, index, 
            index_key, index_pos);
        
        // ---------------------------------------------------------------------
        //  single-thread case
        // ---------------------------------------------------------------------
        if (size == 1) {
            // convert local bucket index into global bins
            index_to_bins(num_keys, N, b, delta, filter, gbeta, bktset, 
                bktpos, index, index_pos, tmp_arr, tmp_pid, tmp_freq, cand_bin, 
                binset, binpos);
#ifdef DEBUG_INFO
            printf("Rank #%d: Signature %2d/%d, #Keys=%d, #Bins=%d\n", rank, 
                i+1, l, num_keys, binpos.size()-1);
#endif
            continue;
        }
        // ---------------------------------------------------------------------
        //  multi-thread case
        // ---------------------------------------------------------------------
        // step 1: convert local bucket index into local buffer
        std::vector<std::vector<int> > buffer;
        int num_buffer = index_to_buffer(num_keys, N, max_buff, lbeta, bktset,
            bktpos, index, index_key, index_pos, tmp_arr, tmp_pid, tmp_freq, 
            cand_bin, buffer);
        
        // step 2: get max round based on num_buffer
        int max_round = get_max_round(size, num_buffer); // start from 1
        
        // step 3: gather & convert all local buffers into candidate list
        std::unordered_map<int, std::vector<std::pair<int,int> > > cand_list;
        std::unordered_map<int, int> cand_cnt;
        for (int j = 0; j < max_round; ++j) {
            // gather all buffers from different local buffers to root
            if (num_buffer < j) buffer.push_back(std::vector<int>());
            int tlen = gather_all_buffers(size, buffer[j], all_buff);

            // @root: convert all buffers into candidate list
            if (rank == 0) {
                all_buff_to_cand_list(tlen, all_buff, cand_list, cand_cnt);
            }
            std::vector<int>().swap(buffer[j]); // clear buffer[j]
        }
        // step 4: convert candidate list into global bins @root
        if (rank == 0) {
            // get keys in ascending order
            std::vector<int> keys(cand_list.size());
            int j = 0;
            for (auto& p : cand_list) keys[j++] = p.first;
            thrust::sort(keys.begin(), keys.end());
            
            for (int key : keys) {
                if (cand_cnt[key] <= b) continue; // skip if |bin| <= b
                
                // get cand_bin from candidate list
                int len = 0; // actual length of cand_bin
                frequent_data(N, gbeta, cand_list[key], tmp_pid, tmp_freq, 
                    cand_bin, len);
                
                // add cand_bin to binset and binpos
                add_cand_bin(len, delta, filter, cand_bin, binset, binpos);
            }
#ifdef DEBUG_INFO
            printf("Rank #%d: Signature %2d/%d, #Keys=%d, #Bins=%d\n", rank, 
                i+1, l, keys.size(), binpos.size()-1);
#endif
            std::vector<int>().swap(keys);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
        std::vector<std::vector<int> >().swap(buffer); // clear buffer
        cand_list.clear();
        cand_cnt.clear();
    }
    delete[] tmp_arr;
    delete[] tmp_pid;
    delete[] tmp_freq;
    delete[] cand_bin;
    
    delete[] index;
    delete[] index_key;
    delete[] index_pos;
    delete[] all_buff;
    
    // broadcast global bins from root to other threads
    if (size > 1) broadcast_set_and_pos(rank, size, binset, binpos);
    
    // accumulate the bin size to get the start position of each bin
    size_t n_binpos = binpos.size();
    for (size_t i = 1; i < n_binpos; ++i) binpos[i] += binpos[i-1];
    
    return binpos.size()-1;
}

// -----------------------------------------------------------------------------
u64 labels_to_index(                // convert labels into index and index_pos
    int   size,                         // number of MPIs (size)
    int   n,                            // number of labels
    int   k,                            // number of centers
    const int *labels,                  // cluster labels for data points
    std::vector<int> &binset,           // bin set (return)
    std::vector<u64> &binpos)           // bin position (return)
{
    assert((u64) n*size < MAX_INT);
    
    int N = n*size; // total num of data points
    int *all_labels = new int[N];
    if (size == 1) {
        // directly copy labels to all labels
        std::copy(labels, labels+n, all_labels);
    }
    else {
        // get all labels from different threads to all threads
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allgather(labels, n, MPI_INT, all_labels, n, MPI_INT, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // sort all labels and its corresponding index
    int *index = new int[N];
    thrust::sequence(index, index+N); // init index with 0,1,2,3,...
    thrust::sort_by_key(all_labels, all_labels+N, index);
    
    // update binset and binpos
    binset.reserve(N);
    binset.insert(binset.end(), index, index+N);
    
    binpos.reserve(k+1); // reserve by the last number of centers
    binpos.push_back(0UL);
    for (int i = 1; i < N; ++i) {
        if (all_labels[i] != all_labels[i-1]) binpos.push_back(i);
    }
    binpos.push_back(N);
    
    // release space
    delete[] index;
    delete[] all_labels;
    
    return binpos.size()-1;
}

// -----------------------------------------------------------------------------
int labels_to_bins(                 // labels to bins & re-number labels
    int rank,                           // MPI rank
    int size,                           // number of MPIs (size)
    int n,                              // number of data points
    int k,                              // number of cluster centers
    int *labels,                        // cluster labels for data (return)
    std::vector<int> &binset,           // bin set (return)
    std::vector<u64> &binpos)           // bin position (return)
{
    std::vector<int>().swap(binset);
    std::vector<u64>().swap(binpos);
    
    // convert labels on local data into global bin set and bin position
    u64 num_bins = labels_to_index(size, n, k, labels, binset, binpos);
    assert(num_bins < MAX_INT);
    
    // re-number labels for local data
    int lb = n * rank; // lower_bound
    int ub = n + lb;   // upper bound
    for (int i = 0; i < num_bins; ++i) {
        const int *bin = &binset[binpos[i]];    // get bin
        int num = int(binpos[i+1] - binpos[i]); // get bin num
        for (int j = 0; j < num; ++j) {
            int id = bin[j];
            if (id >= lb && id < ub) labels[id-lb] = i;
        }
    }
    return num_bins;
}

} // end namespace clustering
