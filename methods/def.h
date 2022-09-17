#pragma once

#include <iostream>
#include <cstdlib>
#include <stdint.h>

namespace clustering {

// -----------------------------------------------------------------------------
//  marco & typedef
// -----------------------------------------------------------------------------
#define SQR(x)              ((x) * (x))
#define DEBUG_INFO

typedef uint8_t  u08;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float    f32;
typedef double   f64;

// -----------------------------------------------------------------------------
//  general constants
// -----------------------------------------------------------------------------
const f32 E                 = 2.7182818F;
const f32 PI                = 3.141592654F;
const f32 FLOAT_ERROR       = 1e-6F;

const f32 MAX_FLOAT         = 3.402823466e+38F;
const f32 MIN_FLOAT         = -MAX_FLOAT;
const u32 UINT32_PRIME      = 4294967291U; // uint32 prime (2^32-5)
const u32 MAX_UINT32        = 4294967295U; // 2^32-1
const int MAX_INT           = 2147483647;  // 2^31-1
const int MIN_INT           = -MAX_INT;

const int RANDOM_SEED       = 666;         // random seed
const int THREAD_NUM        = 32;          // openmp #threads for one node
const int BLOCK_SIZE        = 512;         // block size for cuda
const u64 GPU_MEMORY_LIMIT  = 9UL << 30;   // 9 GB GPU memory limit
const u08 DEVICE_LOCAL_RANK = atoi(std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"));

const int MAX_NUM_HASH      = 1000;        // max #hash functions
const int MAX_BUFF_SIZE     = 1 << 30;     // 2^30 PSM2 upper bound
const int MAX_BIN_SIZE      = 200;         // maximum bin size

const int STATIC_ARRAY_SIZE = 50000000;    // static array size
const int CAND_LEN_SIZE     = 1000000;     // cand len size

// const f32 G_BIN_FREQ     = 0.9f;        // global majority vote freq for bin
// const f32 L_BIN_FREQ     = 0.8f;        // local  majority vote freq for bin
// const f32 G_MODE_FREQ    = 0.5f;        // global majority vote freq for mode
// const f32 L_MODE_FREQ    = 0.2f;        // local  majority vote freq for mode

// -----------------------------------------------------------------------------
struct MPI_INFO {                 // MPI information
    int  rank_;                   // which mpi, start from 0
    int  size_;                   // total number of MPIs
    int  name_len_;               // length of server name
    char name_[100];              // server name
};

} // end namespace clustering
