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
int* read_sparse_data(              // read sparse data from disk
    int   n,                            // number of data points
    u64   *pos,                         // start position of each data point
    const char *fname)                  // input file name
{
    FILE *fp = fopen(fname, "rb");
    if (!fp) { printf("Cannot open %s\n", fname); exit(1); }
    
    printf("read %s\n", fname);
    fread(pos, sizeof(u64), n+1, fp);
    
    int *data = new int[pos[n]];
    fread(data, sizeof(int), pos[n], fp);
    fclose(fp);
    
    return data;
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
    
    int n_each  = (int) std::ceil((double) n / (double) m);
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
    // init parameters
    int  N = atoi(args[1]);
    int  n = atoi(args[2]);
    char dname[50]; strncpy(dname, args[3], sizeof(dname));
    
    // read data set
    char fname[200]; sprintf(fname, "%s_0.bin", dname);
    u64  *pos  = new u64[N+1];
    int  *data = read_sparse_data(N, pos, fname);
    
    // partition data & write them to disk
    write_m_partitions(1, n, dname, pos, data);
    write_m_partitions(2, n, dname, pos, data);
    write_m_partitions(4, n, dname, pos, data);
    write_m_partitions(8, n, dname, pos, data);
    
    delete[] pos;
    delete[] data;
    
    return 0;
}
