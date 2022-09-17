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
int get_record_from_line(           // get a record from a line
    std::string str_data,               // string line
    int d,                              // dimension
    std::vector<int> &record)           // record (return)
{
    std::istringstream ss(str_data);
    int   loc = -1, label = -1, j = 0;
    float val = -1.0f;
    
    while (ss) {
        std::string s;
        if (!getline(ss, s, ' ')) break;
        if (j == 0) {
            label = atoi(s.c_str()); // printf("label = %d\n", label);
        }
        else {
            sscanf(s.c_str(), "%d:%f", &loc, &val);
            // TODO should determine whether we need loc = loc - 1
            loc = loc - 1; assert(loc >= 0 && loc < d);
            record.push_back(loc);
        }
        ++j;
    }
    return j-1; // counter minus 1 for the label
}

// -----------------------------------------------------------------------------
void load(                          // load data and labels into memory
    int   n,                            // number of data points
    int   d,                            // data dimension
    const char *fname,                  // input file
    u64   *pos,                         // start pos for each data (return)
    std::vector<std::vector<int> > &data) // data points (return)
{
    printf("load data from %s\n", fname);
    
    std::ifstream file(fname);
    std::string str;
    int i = 0;
    
    pos[0] = 0UL;
    while (getline(file, str)) {
        pos[i+1] = (u64) get_record_from_line(str, d, data[i]);
        ++i;
        
        if (i%50000000 == 0) printf("i=%d, n=%d\n", i, n);
    }
    assert(i == n);
    file.close();
    
    for (int i = 1; i <= n; ++i) pos[i] += pos[i-1];
    printf("pos[%d]=%lu\n", n, pos[n]);
}

// -----------------------------------------------------------------------------
void copy(                          // copy data
    const std::vector<int> &data,       // original data
    int   *data_flat)                   // destinated data
{
    for (size_t i = 0; i < data.size(); ++i) data_flat[i] = data[i];
}

// -----------------------------------------------------------------------------
void flat_data(                     // flat the data into an array
    int   n,                            // cardinality
    const std::vector<std::vector<int> > &data, // data points
    const u64 *pos,                     // start position of each data point
    int   *data_flat)                   // data_flat (return)
{
    printf("flat data\n");
    for (int i = 0; i < n; ++i) {
        copy(data[i], &data_flat[pos[i]]);
    }
}

// -----------------------------------------------------------------------------
void write_sparse_data(             // write sparse data to disk (binary)
    int   n,                            // cardinality
    const u64 *pos,                     // position of each data
    const int *data,                    // data flat
    const char *fname)                  // output file name
{
    printf("write data to %s\n", fname);
    FILE *fp = fopen(fname, "wb");
    if (!fp) { printf("Could not open %s!!!\n", fname); exit(1); }
    
    fwrite(pos,  sizeof(u64), n+1,    fp);
    fwrite(data, sizeof(int), pos[n], fp);
    fclose(fp);
}

// -----------------------------------------------------------------------------
int* read_sparse_data(              // read sparse data from disk
    int n,                              // number of data points
    u64 *pos,                           // start position of each data point
    const char *fname)                  // input file name
{
    FILE *fp = fopen(fname, "rb");
    if (!fp) { printf("Cannot open %s\n", fname); exit(1); }
    
    fread(pos, sizeof(u64), n+1, fp);
    
    int *data = new int[pos[n]];
    fread(data, sizeof(int), pos[n], fp);
    fclose(fp);
    
    return data;
}

// -----------------------------------------------------------------------------
int main(int nargs, char **args)
{
    clock_t start_time = clock();
    
    int  n = atoi(args[1]);
    int  d = atoi(args[2]);
    char txt_data[200]; strncpy(txt_data, args[3], sizeof(txt_data));
    char bin_data[200]; strncpy(bin_data, args[4], sizeof(bin_data));
    printf("n=%d, d=%d, input=%s, output=%s\n", n, d, txt_data, bin_data);
    create_dir(bin_data);
    
    // load text data
    
    std::vector<std::vector<int> > data(n);
    u64 *pos = new u64[n+1];
    load(n, d, txt_data, pos, data);
    double load_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    printf("load time=%lf Seconds\n\n", load_time);
    
    // convert and write binary data
    int *data_flat = new int[pos[n]];
    flat_data(n, data, pos, data_flat);
    write_sparse_data(n, pos, data_flat, bin_data);
    double total_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    printf("total time=%lf Seconds\n\n", total_time);
    
    // display the first point and the last point for double check
    for (size_t i = pos[0]; i < pos[1]; ++i) {
        printf("%d ", data_flat[i]);
    }
    printf("\n");
    for (size_t i = pos[n-1]; i < pos[n]; ++i) {
        printf("%d ", data_flat[i]);
    }
    printf("\n");
    
    std::vector<std::vector<int> >().swap(data);
    delete[] pos;
    delete[] data_flat;
    
    return 0;
}
