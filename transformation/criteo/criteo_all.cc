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

typedef uint32_t u32;
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
    u32 this_n,                         // number of data points in this file
    u64 N,                              // former shift position
    u64 *pos,                           // store data positions in this file
    int *data,                          // store data in this file
    const char *fname)                  // input file name
{
    FILE *fp = fopen(fname, "rb");
    if (!fp) { printf("Cannot open %s\n", fname); exit(1); }
    
    // read data positions in this file
    fread(pos, sizeof(u64), this_n+1, fp);
    u64 shift = pos[this_n]; // get shift position
    
    // read data
    fread(data, sizeof(int), shift, fp);
    fclose(fp);
    
    for (size_t i = 0; i <= this_n; ++i) pos[i] += N;
    return shift;
}

// -----------------------------------------------------------------------------
void write_sparse_data(             // write sparse data to disk (binary)
    u32   n,                            // cardinality
    const u64 *pos,                     // position of each data
    const int *data,                    // data flat
    const char *fname)                  // output file name
{
    printf("write %s\n", fname);
    FILE *fp = fopen(fname, "wb");
    if (!fp) { printf("Could not open %s!!!\n", fname); exit(1); }
    
    fwrite(pos,  sizeof(u64), n+1,    fp);
    fwrite(data, sizeof(int), pos[n], fp);
    fclose(fp);
}

// -----------------------------------------------------------------------------
int main(int nargs, char **args)
{
    u32 n_array[23] = { 195841983,199563535,196792019,
        181115208,152115810,172548507,204846845,200801003,
        193772492,198424372,185778055,153588700,169003364,
        194216520,194081279,187154596,177984934,163382602,
        142061091,156534237,193627464,192215183,189747893, };
    
    u32 total_n = 4195197692U;    // 592197537U;    // 
    u64 total_N = 102132096134UL; // 14442746080UL; // 
    u64 *pos  = new u64[total_n+1];
    int *data = new int[total_N];
    
    u32  n = 0;
    u64  N = 0;
    char fname[200];
    for (int i = 0; i < 23; ++i) {
        sprintf(fname, "Criteo_Day%d_0.bin", i);
        printf("i=%d, n=%u, N=%lu, fname=%s\n", i, n, N, fname);
        
        u64 shift = read_sparse_data(n_array[i], N, &pos[n], &data[N], fname);
        n += n_array[i];
        N += shift;
    }
    printf("n=%u, total_n=%u, pos[n]=%lu, N=%lu, total_N=%lu\n", n, total_n, 
        pos[n], N, total_N);
    assert(n == total_n && N == total_N);
    
    sprintf(fname, "Criteo_0.bin");
    write_sparse_data(n, pos, data, fname);
    
    printf("The 1b-th data:\n");
    for (size_t i = pos[1000000000U]; i < pos[1000000001U]; ++i) {
        printf("%d ", data[i]);
    }
    printf("\n");
    
    delete[] pos;
    delete[] data;
    
    return 0;
}
