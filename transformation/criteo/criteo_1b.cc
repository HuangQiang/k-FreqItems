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

typedef uint64_t u64;

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
u64 read_sparse_data(               // read sparse data from disk
    int n,                              // number of data points in this file
    u64 N,                              // former shift position
    u64 *pos,                           // store data positions in this file
    int *data,                          // store data in this file
    const char *fname)                  // input file name
{
    FILE *fp = fopen(fname, "rb");
    if (!fp) { printf("Cannot open %s\n", fname); exit(1); }
    
    // read data positions in this file
    fread(pos, sizeof(u64), n+1, fp);
    u64 shift = pos[n]; // get shift position
    
    // read data
    fread(data, sizeof(int), shift, fp);
    fclose(fp);
    
    for (int i = 0; i <= n; ++i) pos[i] += N;
    return shift;
}

// -----------------------------------------------------------------------------
void write_sparse_data(             // write sparse data to disk (binary)
    int   n,                            // cardinality
    const u64 *pos,                     // position of each data
    const int *data,                    // data flat
    const char *fname)                  // output file name
{
    printf("write: n=%d, fname=%s\n", n, fname);
    FILE *fp = fopen(fname, "wb");
    if (!fp) { printf("Could not open %s!!!\n", fname); exit(1); }
    
    fwrite(pos,  sizeof(u64), n+1,    fp);
    fwrite(data, sizeof(int), pos[n], fp);
    fclose(fp);
}

// -----------------------------------------------------------------------------
void init_new_pos(                  // init new data positions
    int   n,                            // number of data points
    u64   N,                            // shift position
    const u64 *pos,                     // old positions
    u64   *new_pos)                     // new positions
{
    for (int i = 0; i <= n; ++i) new_pos[i] = pos[i] - N;
}

// -----------------------------------------------------------------------------
void write_m_partitions(            // write m partitions to disk
    int   m,                            // number of partitions
    int   n,                            // number of data points
    const char *dname,                  // name of dataset
    const u64 *pos,                     // data positions
    const int *data)                    // data points
{
    char prefix[200]; sprintf(prefix, "%d/%s", m, dname);
    create_dir(prefix);
    
    int n_each  = (int) ceil((double) n / (double) m);
    int n_shift = 0;
    for (int i = 0; i < m; ++i) {
        // determine n_each for each partition
        if (n_shift + n_each > n) n_each = n - n_shift;
        
        // init data position for this partition
        u64 *new_pos = new u64[n_each+1];
        init_new_pos(n_each, pos[n_shift], &pos[n_shift], new_pos);
        
        // write this partition
        char fname[200]; sprintf(fname, "%s_%d.bin", prefix, i);
        write_sparse_data(n_each, new_pos, &data[pos[n_shift]], fname);
        
        // update n_shift and release space
        n_shift += n_each;
        delete[] new_pos;
    }
    assert(n_shift == n);
}

// -----------------------------------------------------------------------------
int main(int nargs, char **args)
{
    int n_arr[6] = { 195841983,199563535,196792019,181115208,152115810,
        172548507, };
    
    int total_n = 1097977062;
    u64 total_N = 42820884591UL;
    u64 *pos  = new u64[total_n+1];
    int *data = new int[total_N];
    
    int  n = 0;
    u64  N = 0;
    char fname[200];
    
    for (int i = 0; i < 6; ++i) {
        sprintf(fname, "Criteo_Day%d_0.bin", i);
        printf("i=%d, n=%d, N=%lu, fname=%s\n", i, n, N, fname);
        
        u64 shift = read_sparse_data(n_arr[i], N, &pos[n], &data[N], fname);
        n += n_arr[i];
        N += shift;
    }
    printf("n=%d, total_n=%d, pos[n]=%lu, N=%lu, total_N=%lu\n", n, total_n, 
        pos[n], N, total_N);
    assert(n == total_n && N == total_N);
    
    // extract 10 million data & write it to disk
    n = 10000000;
    sprintf(fname, "Criteo10M");
    write_m_partitions(1, n, fname, pos, data);
    write_m_partitions(2, n, fname, pos, data);
    write_m_partitions(4, n, fname, pos, data);
    write_m_partitions(8, n, fname, pos, data);
    
    // extract 1 billion data & write it to disk
    n = 1000000000;
    sprintf(fname, "Criteo1B");
    write_m_partitions(1, n, fname, pos, data);
    write_m_partitions(2, n, fname, pos, data);
    write_m_partitions(4, n, fname, pos, data);
    write_m_partitions(8, n, fname, pos, data);
    
    delete[] pos;
    delete[] data;
    
    return 0;
}
