#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <stdint.h>
#include <unistd.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>

timeval g_start_time;
timeval g_end_time;

typedef uint8_t  u08;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float    f32;
typedef double   f64;

const f32 MAX_FLOAT    = 3.402823466e+38F;
const f32 MIN_FLOAT    = -MAX_FLOAT;
const u32 UINT32_PRIME = 4294967291U; // uint32 prime (2^32-5)
const u32 MAX_UINT32   = 4294967295U; // 2^32-1
const int MAX_INT      = 2147483647;  // 2^31-1
const int MIN_INT      = -MAX_INT;

const f32 E            = 2.7182818F;
const f32 PI           = 3.141592654F;
const f32 FLOAT_ERROR  = 1e-6F;
const int RANDOM_SEED  = 666;

// -----------------------------------------------------------------------------
void create_dir(                    // create dir if the path not exists
    char *path)                         // input path
{
    int len = (int) strlen(path);
    for (int i = 0; i < len; ++i) {
        if (path[i] != '/') continue;
        
        char ch = path[i+1]; path[i+1] = '\0';
        if (access(path, F_OK) != 0) { // create directory if not exists
            if (mkdir(path, 0755) != 0) {
                printf("Could not create directory %s\n", path); exit(1);
            }
        }
        path[i+1] = ch;
    }
}

// -----------------------------------------------------------------------------
int* read_sparse_data(              // read sparse data from disk
    int   n,                            // number of data points
    u64   *datapos,                     // data position
    const char *fname)                  // input file name
{
    FILE *fp = fopen(fname, "rb");
    if (!fp) { printf("Cannot open %s\n", fname); exit(1); }
    
    fread(datapos, sizeof(u64), n+1, fp);
    
    int *dataset = new int[datapos[n]];
    fread(dataset, sizeof(int), datapos[n], fp);
    fclose(fp);
    
    printf("read data set from %s\n\n", fname);
    return dataset;
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
void generate_distinct_ids(         // generate m distinct ids
    int n,                              // total range
    int m,                              // m value
    int times,                          // repeated times
    int *total_distinct_ids)            // total distinct ids (return)
{
    bool *select = new bool[n];
    int  id = -1;
    
    for (int i = 0; i < times; ++i) {
        int *distinct_ids = total_distinct_ids + i*m;
        
        memset(select, false, sizeof(bool)*n);
        for (int j = 0; j < m; ++j) {
            // every time draw a distinct id uniformly at from from [0,n-1]
            do { id = uniform_u32(0, n-1); } while (select[id]);
            
            select[id] = true;
            distinct_ids[j] = id;
        }
    }
    delete[] select;
}

// -----------------------------------------------------------------------------
int get_length(                     // get the length of pos
    int   id,                           // input id
    const u64 *pos)                     // pos array
{
    return int(pos[id+1] - pos[id]);
}

// -----------------------------------------------------------------------------
float jaccard_dist(                 // calc jaccard distance
    int   n_data,                       // number of data dimensions
    int   n_mode,                       // number of mode dimensions
    const int *data,                    // data point
    const int *mode)                    // mode
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
int distinct_coord_and_freq(        // get max freq, distinct coords & freqs
    int total_num,                      // total number of coordinates
    int *arr,                           // store all coordinates (allow modify)
    int *coord,                         // distinct coordinates (return)
    int *freq,                          // frequency (return)
    int &cnt)                           // counter for #distinct (return)
{
    // sort all coordinates in ascending order
    std::sort(arr, arr + total_num);
    
    // get the distinct coordinates and their frequencies (sequential)
    cnt = 0;
    int max_freq = 0, last = 0, this_freq = -1;
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
void calc_stat(                     // calc stat between center and cluster
    int   m,                            // a cluster of m distinct ids
    int   len,                          // length of center
    const int *center,                  // input center
    const int *distinct_ids,            // m distinct ids
    const int *dataset,                 // data set
    const u64 *datapos,                 // data position
    float &mse,                         // mean square error
    float &mae)                         // mean absolute error
{
    mse = 0.0f; mae = 0.0f;
    for (int i = 0; i < m; ++i) {
        int did  = distinct_ids[i];
        int dlen = get_length(did, datapos);
        const int *data = dataset + datapos[did];
        
        float dist = jaccard_dist(dlen, len, data, center);
        mse += dist*dist;
        mae += dist;
    }
    mse /= m;
    mae /= m;
}

// -----------------------------------------------------------------------------
void freqitems(                     // consider frequent items (mode) as center
    int   m,                            // m distinct ids
    float step,                         // step size for alpha
    const int *distinct_ids,            // distinct ids
    const int *dataset,                 // data set
    const u64 *datapos,                 // data position
    const char *fname)                  // output file name
{
    timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    
    printf("Write FreqItem to %s\n", fname);
    FILE *fp = fopen(fname, "w");
    if (!fp) { printf("Cannot open %s\n", fname); exit(1); }
    
    // get the total number of coordinates (total_num) in this cluster
    int id, total_num = 0;
    for (int i = 0; i < m; ++i) {
        id = distinct_ids[i]; 
        total_num += get_length(id, datapos);
    }
    
    // init an array to store all coordinates in this cluster
    int *arr = new int[total_num];
    int cnt = 0, len = -1;
    for (int i = 0; i < m; ++i) {
        id  = distinct_ids[i];
        len = get_length(id, datapos);
        const int *data = dataset + datapos[id];
        
        std::copy(data, data+len, arr+cnt);
        cnt += len;
    }
    assert(cnt == total_num);
    
    // get the distinct coordinates and their frequencies
    int *coord = new int[total_num];
    int *freq  = new int[total_num];
    int num = 0; // number of distinct coordinates
    int max_freq = distinct_coord_and_freq(total_num, arr, coord, freq, num);
    
    int    *center = new int[num];
    float  mse  = -1.0f, mae = -1.0f;
    double time = -1.0;
    
    // choose the frequent items as center
    float  alpha = 0.0f;
    float  best_alpha = -1.0f, min_mse = MAX_FLOAT, min_mae = MAX_FLOAT;
    double best_time  = -1.0;
    
    gettimeofday(&end_time, NULL);
    
    double pre_time  = end_time.tv_sec - start_time.tv_sec + (end_time.tv_usec -
        start_time.tv_usec) / 1000000.0;
    double last_time = pre_time;
    
    fprintf(fp, "alpha,MSE,MAE,Time\n");
    while (alpha < 1.000001f) {
        // check for every alpha
        float threshold = (int) ceil((float) max_freq*alpha);
        if (threshold > max_freq) threshold = max_freq;
        
        cnt = 0; // length of center
        for (int i = 0; i < num; ++i) {
            if (freq[i] >= threshold) center[cnt++] = coord[i];
        }
        calc_stat(m, cnt, center, distinct_ids, dataset, datapos, mse, mae);
        gettimeofday(&end_time, NULL);
        
        double runtime  = end_time.tv_sec - start_time.tv_sec + 
            (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
        time = pre_time + runtime - last_time;
        fprintf(fp, "%.2f,%f,%f,%lf\n", alpha, mse, mae, time);
        
        // update statistics
        if (mse < min_mse) { 
            best_alpha = alpha; 
            min_mse = mse; min_mae = mae; best_time = time;
        }
        alpha += step;
        last_time = runtime;
    }
    fprintf(fp, "\nFreqItem (alpha = %.2f)\n", best_alpha);
    fprintf(fp, "FreqItem,%f,%f,%lf\n\n", min_mse, min_mae, best_time);
    fclose(fp);
    
    // release space
    delete[] arr;
    delete[] coord;
    delete[] freq;
    delete[] center;
}

// -----------------------------------------------------------------------------
void mode1(                         // consider mode1 as center
    int   m,                            // m distinct ids
    const int *distinct_ids,            // distinct ids
    const int *dataset,                 // data set
    const u64 *datapos,                 // data position
    const char *fname)                  // output file name
{
    timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    
    printf("Write Mode1 to %s\n", fname);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Cannot open %s\n", fname); exit(1); }
    
    // get the total number of coordinates (total_num) in this cluster
    int id, total_num = 0;
    for (int i = 0; i < m; ++i) {
        id = distinct_ids[i]; 
        total_num += get_length(id, datapos);
    }
    
    // init an array to store all coordinates in this cluster
    int *arr = new int[total_num];
    int cnt = 0, len = -1;
    for (int i = 0; i < m; ++i) {
        id  = distinct_ids[i];
        len = get_length(id, datapos);
        const int *data = dataset + datapos[id];
        
        std::copy(data, data+len, arr+cnt);
        cnt += len;
    }
    assert(cnt == total_num);
    
    // get the distinct coordinates and their frequencies
    int *coord = new int[total_num];
    int *freq  = new int[total_num];
    int num = 0; // number of distinct coordinates
    int max_freq = distinct_coord_and_freq(total_num, arr, coord, freq, num);
    
    // choose every distinct coordinates (mode1) as center
    int *center = new int[num];
    float mse = -1.0f, mae = -1.0f;
    for (int i = 0; i < num; ++i) {
        center[i] = coord[i];
    }
    calc_stat(m, num, center, distinct_ids, dataset, datapos, mse, mae);
    gettimeofday(&end_time, NULL);
    double runtime  = end_time.tv_sec - start_time.tv_sec + (end_time.tv_usec - 
        start_time.tv_usec) / 1000000.0;
    
    fprintf(fp, "Mode (each dim)\n");
    fprintf(fp, "Mode1,%f,%f,%lf\n\n", mse, mae, runtime);
    fclose(fp);
    
    // release space
    delete[] arr;
    delete[] coord;
    delete[] freq;
    delete[] center;
}

// -----------------------------------------------------------------------------
void mode2(                         // consider mode2 as center
    int   m,                            // m distinct ids
    const int *distinct_ids,            // distinct ids
    const int *dataset,                 // data set
    const u64 *datapos,                 // data position
    const char *fname)                  // output file name
{
    timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    
    printf("Write Mode2 to %s\n", fname);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Cannot open %s\n", fname); exit(1); }
    
    // get the total number of coordinates (total_num) in this cluster
    int id, total_num = 0;
    for (int i = 0; i < m; ++i) {
        id = distinct_ids[i]; 
        total_num += get_length(id, datapos);
    }
    
    // init an array to store all coordinates in this cluster
    int *arr = new int[total_num];
    int cnt = 0, len = -1;
    for (int i = 0; i < m; ++i) {
        id  = distinct_ids[i];
        len = get_length(id, datapos);
        const int *data = dataset + datapos[id];
        
        std::copy(data, data+len, arr+cnt);
        cnt += len;
    }
    assert(cnt == total_num);
    
    // get the distinct coordinates and their frequencies
    int *coord = new int[total_num];
    int *freq  = new int[total_num];
    int num = 0; // number of distinct coordinates
    int max_freq = distinct_coord_and_freq(total_num, arr, coord, freq, num);
    
    // choose the coordinates with max freq (mode2) as center
    int *center = new int[num];
    float mse = -1.0f, mae = -1.0f;
    cnt = 0; // length of center
    for (int i = 0; i < num; ++i) {
        if (freq[i] >= max_freq) center[cnt++] = coord[i];
    }
    calc_stat(m, cnt, center, distinct_ids, dataset, datapos, mse, mae);
    gettimeofday(&end_time, NULL);
    double runtime  = end_time.tv_sec - start_time.tv_sec + (end_time.tv_usec - 
        start_time.tv_usec) / 1000000.0;
    
    fprintf(fp, "Mode (all dim)\n");
    fprintf(fp, "Mode2,%f,%f,%lf\n\n", mse, mae, runtime);
    fclose(fp);
    
    // release space
    delete[] arr;
    delete[] coord;
    delete[] freq;
    delete[] center;
}

// -----------------------------------------------------------------------------
void medoid(                        // consider actual data object as center
    int   m,                            // m distinct ids
    const int *distinct_ids,            // distinct ids
    const int *dataset,                 // data set
    const u64 *datapos,                 // data position
    const char *fname)                  // output file name
{
    timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    
    printf("Write Medoid to %s\n\n", fname);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Cannot open %s\n", fname); exit(1); }
    
    // choose different data objects as center for a cluster
    int   min_num = -1;
    float min_mse = MAX_FLOAT, min_mae = MAX_FLOAT;
    float *mse = new float[m]; 
    float *mae = new float[m];
    
    for (int i = 0; i < m; ++i) {
        // get the center
        int id  = distinct_ids[i];
        int len = get_length(id, datapos);
        const int *center = dataset + datapos[id];
        
        // calc mse and mae between the center and the cluster
        calc_stat(m, len, center, distinct_ids, dataset, datapos, mse[i], mae[i]);
        if (mse[i] < min_mse) { 
            min_mse = mse[i]; min_mae = mae[i]; min_num = i+1;
        }
    }
    gettimeofday(&end_time, NULL);
    double runtime  = end_time.tv_sec - start_time.tv_sec + (end_time.tv_usec - 
        start_time.tv_usec) / 1000000.0;
    fprintf(fp, "Medoid (ID=%d)\n", min_num);
    fprintf(fp, "Medoid,%f,%f,%lf\n\n", min_mse, min_mae, runtime);
    
    fprintf(fp, "ID,MSE,MAE,DataID\n");
    for (int i = 0; i < m; ++i) {
        fprintf(fp, "%d,%f,%f,%d\n", i+1, mse[i], mae[i], distinct_ids[i]);
    }
    fclose(fp);
    
    delete[] mse;
    delete[] mae;
}

// -----------------------------------------------------------------------------
int main(int nargs, char **args)
{
    srand(0);
    
    int   n     = atoi(args[1]);
    int   d     = atoi(args[2]);
    int   m     = atoi(args[3]);
    int   times = atoi(args[4]);
    float step  = 0.01f;
    char input[200];  strncpy(input,  args[5], sizeof(input));
    char output[200]; strncpy(output, args[6], sizeof(output));
    create_dir(output);
    
    printf("n      = %d\n", n);
    printf("d      = %d\n", d);
    printf("m      = %d\n", m);
    printf("times  = %d\n", times);
    printf("step   = %.2f\n", step);
    printf("input  = %s\n", input);
    printf("output = %s\n", output);
    printf("\n");
    
    // read sparse data
    u64 *datapos = new u64[n+1];
    int *dataset = read_sparse_data(n, datapos, input);
    
    // randomly select m distinct data objects as a cluster
    int *total_distinct_ids = new int[m*times];
    generate_distinct_ids(n, m, times, total_distinct_ids);
    
    for (int i = 0; i < times; ++i) {
        const int *distinct_ids = (const int*) total_distinct_ids + i*m;
        for (int j = 0; j < 10; ++j) printf("%d ", distinct_ids[j]);
        printf("\n");
    }
    
    for (int i = 0; i < times; ++i) {
        const int *distinct_ids = (const int*) total_distinct_ids + i*m;
        char fname[200]; sprintf(fname, "%s_%d_%d.csv", output, m, i+1);
        
        // case 1: consider frequent items as center
        freqitems(m, step, distinct_ids, dataset, datapos, fname);
    
        // case 2: consider mode1 as center
        mode1(m, distinct_ids, dataset, datapos, fname);
        
        // case 3: consider mode2 as center
        mode2(m, distinct_ids, dataset, datapos, fname);
        
        // case 4: consider actual data object (medoid) as center
        medoid(m, distinct_ids, dataset, datapos, fname);
    }
    // release space
    delete[] datapos;
    delete[] dataset;
    delete[] total_distinct_ids;
    
    return 0;
}
