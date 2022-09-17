#include "util.cuh"

namespace clustering {

int g_k               = -1;         // global param: #clusters
f32 g_mae             = -1.0f;      // global param: mean absolute error
f32 g_mse             = -1.0f;      // global param: mean square   error
f64 g_tot_wc_time     = -1.0;       // global param: total wall clock time (s)
f64 g_tot_cpu_time    = -1.0;       // global param: total cpu time (s)

int g_num_buckets     = -1;         // global param: local number of buckets
int g_tot_buckets     = -1;         // global param: total number of buckets
int g_tot_bins        = -1;         // global param: total number of bins
int g_tot_seeds       = -1;         // global param: total number of seeds
f64 g_phase1_wc_time  = -1.0;       // global param: phase 1 wall clock time (s)
f64 g_phase2_wc_time  = -1.0;       // global param: phase 2 wall clock time (s)
f64 g_phase3_wc_time  = -1.0;       // global param: phase 3 wall clock time (s)
f64 g_eval_wc_time    = -1.0;       // global param: eval wall clock time (s)
f64 g_eval_cpu_time   = -1.0;       // global param: eval cpu time (s)
f64 g_silk_wc_time    = -1.0;       // global param: silk wall clock time (s)
f64 g_silk_cpu_time   = -1.0;       // global param: silk cpu time (s)

int g_iter            = -1;         // global param: #iterations
f64 g_init_wc_time    = -1.0;       // global param: init wall clock time (s)
f64 g_init_cpu_time   = -1.0;       // global param: init cpu time (s)
f64 g_iter_wc_time    = -1.0;       // global param: iter wall clock time (s)
f64 g_iter_cpu_time   = -1.0;       // global param: iter cpu time (s)
f64 g_kfreqitems_wc_time  = -1.0;   // global param: k-freqitems wall clock time (s)
f64 g_kfreqitems_cpu_time = -1.0;   // global param: k-freqitems cpu time (s)

// -----------------------------------------------------------------------------
void create_dir(                    // create dir if the path does not exist
    char *path)                         // input path
{
    int len = (int) strlen(path);
    for (int i = 0; i < len; ++i) {
        if (path[i] != '/') continue;
        
        char ch = path[i+1]; path[i+1] = '\0';
        if (access(path, F_OK) != 0) { // create the directory if not exist
            if (mkdir(path, 0755) != 0) {
                printf("Could not create %s\n", path); exit(1);
            }
        }
        path[i+1] = ch;
    }
}

// -----------------------------------------------------------------------------
void init_mpi_comm(                 // initialize mpi communication
    MPI_INFO &mpi_info)                 // mpi_info (return)
{
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_info.rank_);
    MPI_Get_processor_name(mpi_info.name_, &mpi_info.name_len_);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_info.size_);

    printf("-----------------------------------------------------------------\n"
        "Rank #%d: Initialize MPI with %s; Comunication starts\n\n", 
        mpi_info.rank_, mpi_info.name_);
}

// -----------------------------------------------------------------------------
void finalize_mpi_comm(             // finalize mpi communication
    const MPI_INFO &mpi_info)           // mpi_info
{
    MPI_Finalize();

    printf("Rank #%d: Finalize MPI with %s; Communication ends\n" 
        "-----------------------------------------------------------------\n\n",
        mpi_info.rank_, mpi_info.name_);
}

// -----------------------------------------------------------------------------
void copy_pos(                      // copy a partial datapos to another pos
    int   n,                            // length of partial datapos
    const u64* datapos,                 // partial datapos
    u64   *pos)                         // another pos (return)
{
    u64 start = datapos[0];
    for (int j = 0; j < n; ++j) pos[j] = datapos[j] - start;
}

// -----------------------------------------------------------------------------
void all_buff_to_cand_list(         // convert all buffers into cand list
    int   n,                            // length of all buffers
    const int *all_buff,                // all buffers
    std::unordered_map<int, std::vector<std::pair<int, int> > > &cand_list)
{
    int i = 0;
    while (i < n) {
        int   key = all_buff[i++];
        int   len = all_buff[i++];
        const int *cand = &all_buff[i]; i += len;
        const int *freq = &all_buff[i]; i += len;
        
        // Note: the value of cand_list[key] may be duplicated
        std::vector<std::pair<int, int> > &tmp = cand_list[key];
        tmp.reserve(tmp.size() + len);
        for (int j = 0; j < len; ++j) {
            tmp.push_back(std::make_pair(cand[j], freq[j]));
        }
    }
    assert(i == n);
}

// -----------------------------------------------------------------------------
void all_buff_to_cand_list(         // convert all buffers into cand list
    int   n,                            // length of all buffers
    const int *all_buff,                // all buffers
    std::unordered_map<int, std::vector<std::pair<int, int> > > &cand_list,
    std::unordered_map<int, int> &cand_cnt)
{
    int i = 0;
    while (i < n) {
        int   key  = all_buff[i++]; // get bin key
        int   num  = all_buff[i++]; // get bin num
        int   freq = all_buff[i++]; // get max freq
        int   len  = all_buff[i++]; // get number of candidates
        const int *cand = &all_buff[i]; i += len;
        
        // Note: the value of cand_list[key] may be duplicated
        std::vector<std::pair<int, int> > &tmp = cand_list[key];
        if (cand_cnt[key] < 0) continue;
        
        int new_size = tmp.size() + len;
        if (new_size > CAND_LEN_SIZE) {
            // printf("new_size=%d, cand_cnt=%d\n", new_size, cand_cnt[key]);
            cand_cnt[key] = -1;
        }
        else {
            tmp.reserve(new_size);
            for (int j = 0; j < len; ++j) {
                tmp.push_back(std::make_pair(cand[j], freq));
            }
            cand_cnt[key] += num; // update bucket counter
        }
    }
}

// -----------------------------------------------------------------------------
u64 get_total_coords(               // get total number of coordinates
    int size,                           // number of MPIs (size)
    u64 num_coords)                     // number of coordinates (local)
{
    // single-thread case
    if (size == 1) return num_coords;
    
    // multi-thread case
    u64 *r_num_coords = new u64[size];
    memset(r_num_coords, 0UL, sizeof(u64)*size); // init 0

    // broadcast num_coords to different threads
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allgather(&num_coords, 1, MPI_UINT64_T, r_num_coords, 1, MPI_UINT64_T, 
        MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // sum up the r_num_coords array
    u64 tot_coords = 0UL;
    for (int i = 0; i < size; ++i) tot_coords += r_num_coords[i];
    delete[] r_num_coords;
    
    return tot_coords;
}

// -----------------------------------------------------------------------------
int get_total_buckets(              // get total number of buckets
    int size,                           // number of MPIs (size)
    int num_buckets)                    // number of buckets (local)
{
    // single-thread case: g_tot_buckets == g_num_buckets
    if (size == 1) return num_buckets;
    
    // multi-thread case: broadcast num_buckets to different threads
    int *r_num_buckets = new int[size];
    memset(r_num_buckets, 0, sizeof(int)*size); // init 0

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allgather(&num_buckets, 1, MPI_INT, r_num_buckets, 1, MPI_INT, 
        MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // sum up the r_num_buckets array
    u64 tot_buckets = 0UL;
    for (int i = 0; i < size; ++i) tot_buckets += r_num_buckets[i];
    delete[] r_num_buckets;
    
    if (tot_buckets > MAX_INT) exit(1); // only support range int
    return tot_buckets;
}

// -----------------------------------------------------------------------------
int get_max_round(                  // broadcast num_buffer to get max round
    int size,                           // number of MPIs (size)
    int num_buffer)                     // number of buffer (local)
{
    // single-thread case
    if (size == 1) return num_buffer + 1;
    
    // multi-thread case
    int *r_num_buffer = new int[size];
    for (int i = 0; i < size; ++i) r_num_buffer[i] = 0;

    // broadcast num_buffer to different threads
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allgather(&num_buffer, 1, MPI_INT, r_num_buffer, 1, MPI_INT, 
        MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // get max_round from the array r_num_buffer
    int max_round = r_num_buffer[0];
    for (int i = 1; i < size; ++i) {
        if (r_num_buffer[i] > max_round) max_round = r_num_buffer[i];
    }
    delete[] r_num_buffer;
    
    return max_round + 1; // start from 1
}

// -----------------------------------------------------------------------------
int gather_all_buffers(             // gather buffers from diff threads to root
    int   size,                         // number of MPIs (size)
    const std::vector<int> &buffer,     // buffer in local
    int   *all_buff)                    // all buffers at root (return)
{
    // gather the length of different buffers from different threads
    int len = (int) buffer.size(); // get the length of this buffer
    int *rlen = new int[size];
    for (int i = 0; i < size; ++i) rlen[i] = 0;
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(&len, 1, MPI_INT, rlen, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // get the total length of buffer (tlen) from all threads
    int tlen = 0;
    for (int i = 0; i < size; ++i) tlen += rlen[i];
    assert(tlen <= MAX_BUFF_SIZE);
    
    // init displacements to gather buffers from different threads to root
    int *displs = new int[size];
    displs[0] = 0;
    for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + rlen[i-1];

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(buffer.data(), len, MPI_INT, all_buff, rlen, displs, MPI_INT, 
        0, MPI_COMM_WORLD); // 0 is the root location
    MPI_Barrier(MPI_COMM_WORLD);
    
    delete[] rlen;
    delete[] displs;
    
    return tlen;
}

// -----------------------------------------------------------------------------
void broadcast_set_and_pos(         // broadcast set and pos to other threads
    int rank,                           // MPI rank
    int size,                           // number of MPIs (size)
    std::vector<int> &binset,           // bin set (return)
    std::vector<u64> &binpos)           // bin position (return)
{
    // broadcast binpos from root to other threads
    u64 binpos_size = binpos.size();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&binpos_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank > 0) binpos.resize(binpos_size);
    
    assert(binpos_size <= MAX_INT);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(binpos.data(), binpos_size, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // broadcast binset from root to other threads
    u64 binset_size = binset.size();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&binset_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank > 0) binset.resize(binset_size);
    
    int batch = MAX_BUFF_SIZE;
    for (size_t i = 0; i < binset_size; i += batch) {
        if (i+batch > binset_size) batch = binset_size-i;

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(binset.data()+i, batch, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// -----------------------------------------------------------------------------
u32 uniform_u32(                    // gen a random variable from uniform u32
    u32 min,                            // min value
    u32 max)                            // max value
{
    u32 r = 0U;
    if (RAND_MAX >= max - min) {
        r = min + (u32) ((max-min+1.0)*rand() / (RAND_MAX+1.0));
    }
    else {
        r = min + (u32) ((max-min+1.0) * 
            ((u64) rand() * ((u64) RAND_MAX+1.0) + (u64) rand()) / 
            ((u64) RAND_MAX*((u64) RAND_MAX+1.0) + (u64) RAND_MAX+1.0));
    }
    assert(r >= min && r <= max);
    return r; 
}

// -----------------------------------------------------------------------------
float uniform(                      // gen a random variable from uniform distr.
    float start,                        // start position
    float end)                          // end position
{
    assert(start < end);
    return start + ((end-start)*rand() / (float) RAND_MAX);
}

// -----------------------------------------------------------------------------
float gaussian()                    // gen a random variable from N(0,1)
{
    float x1 = -1.0f;
    do {
        x1 = uniform(0.0f, 1.0f);
    } while (x1 < FLOAT_ERROR); // cannot take log 0 for x1
    
    float x2 = uniform(0.0f, 1.0f);
    return sqrt(-2.0f*log(x1)) * cos(2.0f*PI*x2);
}

// -----------------------------------------------------------------------------
float cauchy()                      // gen a random variable from Cauchy(1,0)
{
    float x = gaussian();
    float y = gaussian();
    if (fabs(y) < FLOAT_ERROR) y = FLOAT_ERROR;
    
    return x / y;
}

// -----------------------------------------------------------------------------
void syn_hash_params(               // synchronize hash parameters
    int   len_s,                        // length of random projection
    int   len_p,                        // length of random shift
    float *shift,                       // random shift (return)
    float *proj)                        // random projection (return)
{
    // C int      --> MPI_INT
    // C float    --> MPI_FLOAT
    // C double   --> MPI_DOUBLE
    // C uint32_t --> MPI_UINT32_T
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(shift, len_s, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(proj,  len_p, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
}

// -----------------------------------------------------------------------------
void output_buckets(                // output buckets to disk
    int   rank,                         // mpi rank
    int   n,                            // number of buckets
    const int  *bktset,                 // bucket set
    const u64  *bktpos,                 // bucket position
    const char *prefix)                 // prefix path
{
    char fname[100]; sprintf(fname, "%s_buckets_%d", prefix, rank);
    FILE *fp = fopen(fname, "wb");
    if (!fp) { printf("Could not open %s\n", fname); exit(1); }

    fwrite(&n,     sizeof(int), 1,         fp);
    fwrite(bktpos, sizeof(u64), n+1,       fp);
    fwrite(bktset, sizeof(int), bktpos[n], fp);
    fclose(fp);
}

// -----------------------------------------------------------------------------
void output_bins(                   // output bins to disk
    int   rank,                         // mpi rank
    int   n,                            // number of bins
    const int  *binset,                 // bin set
    const u64  *binpos,                 // bin position
    const char *prefix)                 // prefix path
{
    if (rank > 0) return;
    
    char fname[100]; sprintf(fname, "%s_bins_%d", prefix, rank);
    FILE *fp = fopen(fname, "wb");
    if (!fp) { printf("Could not open %s\n", fname); exit(1); }

    fwrite(&n,     sizeof(int), 1,         fp);
    fwrite(binpos, sizeof(u64), n+1,       fp);
    fwrite(binset, sizeof(int), binpos[n], fp);
    fclose(fp);
}

// -----------------------------------------------------------------------------
void output_centroids(              // output centroids to disk
    int   rank,                         // mpi rank
    int   len,                          // len = k * d
    const float *centroids,             // centroids
    const char  *prefix)                // prefix
{
    if (rank > 0) return;

    char fname[100]; sprintf(fname, "%s_centroids_%d.bin", prefix, rank);
    FILE *fp = fopen(fname, "wb");
    if (!fp) { printf("Could not open %s\n", fname); exit(1); }
    
    fwrite(centroids, sizeof(float), len, fp);
    fclose(fp);
}

} // end namespace clustering
