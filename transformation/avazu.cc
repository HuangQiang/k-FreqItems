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
    const char *fname,                  // input file name
    uint64_t *pos,                      // start pos for each data (return)
    std::vector<std::vector<int> > &data) // data points (return)
{
    printf("load data from %s\n", fname);
    
    std::ifstream file(fname);
    std::string str;
    int i = 0;
    
    while (getline(file, str)) {
        pos[i+1] = (uint64_t) get_record_from_line(str, d, data[i]);
        ++i;
    }
    assert(i == n);
    file.close();
    
    for (int i = 1; i <= n; ++i) pos[i] += pos[i-1];
    printf("pos[%d] = %lu\n", n, pos[n]);
}

// -----------------------------------------------------------------------------
void copy(                          // copy data
    const std::vector<int> &data,       // original data
    int   *data_flat)                   // destinated data
{
    for (int i = 0; i < data.size(); ++i) data_flat[i] = data[i];
}

// -----------------------------------------------------------------------------
void flat_data(                     // flat the data into an array
    int   n,                            // cardinality
    const std::vector<std::vector<int> > &data, // data points
    const uint64_t *pos,                // start position of each data point
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
    const uint64_t *pos,                // position of each data
    const int  *data,                   // data flat
    const char *fname)                  // output file name
{
    printf("write data to %s\n", fname);
    FILE *fp = fopen(fname, "wb");
    if (!fp) { printf("Could not open %s!!!\n", fname); exit(1); }

    fwrite(pos,  sizeof(uint64_t), n+1,    fp);
    fwrite(data, sizeof(int),      pos[n], fp);
    fclose(fp);
}

// -----------------------------------------------------------------------------
int* read_sparse_data(              // read sparse data from disk
    int n,                              // number of data points
    uint64_t *pos,                      // start position of each data point
    const char *fname)                  // input file name
{
    FILE *fp = fopen(fname, "rb");
    if (!fp) { printf("Cannot open %s\n", fname); exit(1); }
    
    fread(pos, sizeof(uint64_t), n+1, fp);
    
    int *data = new int[pos[n]];
    fread(data, sizeof(int), pos[n], fp);
    fclose(fp);
    
    return data;
}

// -----------------------------------------------------------------------------
int main(int nargs, char **args)
{
    int  n1 = atoi(args[1]);
    int  n2 = atoi(args[2]);
    int  d  = atoi(args[3]);
    int  N  = n1 + n2;
    char input1[200]; strncpy(input1, args[4], sizeof(input1));
    char input2[200]; strncpy(input2, args[5], sizeof(input2));
    char output[200]; strncpy(output, args[6], sizeof(output));
    create_dir(output);
    
    std::vector<std::vector<int> > data1(n1);
    std::vector<std::vector<int> > data2(n2);
    uint64_t *pos = new uint64_t[N+1];
    pos[0] = 0UL;
    load(n1, d, input1, pos, data1);
    load(n2, d, input2, pos+n1, data2);
    printf("pos[%d] = %lu\n", N, pos[N]);
    
    int *data_flat = new int[pos[N]];
    flat_data(n1, data1, pos, data_flat);
    flat_data(n2, data2, pos+n1, data_flat);
    
    write_sparse_data(N, pos, data_flat, output);
    
    // uint64_t *pos = new uint64_t[n+1];
    // int *data_flat = read_sparse_data(n, pos, output);
    
    for (uint64_t i = pos[0]; i < pos[1]; ++i) {
        printf("%d ", data_flat[i]);
    }
    printf("\n");
    for (uint64_t i = pos[n1]; i < pos[n1+1]; ++i) {
        printf("%d ", data_flat[i]);
    }
    printf("\n");
    for (uint64_t i = pos[N-1]; i < pos[N]; ++i) {
        printf("%d ", data_flat[i]);
    }
    printf("\n");
    
    delete[] pos;
    delete[] data_flat;
    std::vector<std::vector<int> >().swap(data1);
    std::vector<std::vector<int> >().swap(data2);
    
    return 0;
}
