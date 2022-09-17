#include "eval.h"

namespace clustering {

// -----------------------------------------------------------------------------
void calc_global_stat(              // calc global statistics
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   k,                            // number of clusters
    float *c_mae,                       // mean absolute error (allow modify)
    float *c_mse,                       // mean square   error (allow modify)
    float &mae,                         // mean absolute error (return)
    float &mse)                         // mean square error (return)
{
    f32 *all_mae = new f32[k*size];
    f32 *all_mse = new f32[k*size];
    
    if (size == 1) {
        // single-thread case: directly copy one to another
        std::copy(c_mae, c_mae+k, all_mae);
        std::copy(c_mse, c_mse+k, all_mse);
    }
    else {
        // multi-thread case: gather to all threads
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allgather(c_mae, k, MPI_FLOAT, all_mae, k, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allgather(c_mse, k, MPI_FLOAT, all_mse, k, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // init 0 for computation
    mae = 0.0f; mse = 0.0f;
    for (int i = 0; i < k; ++i) {
        f32 this_mae = 0.0f;
        f32 this_mse = 0.0f;
        
        for (int j = 0; j < size; ++j) {
            int id = j*k + i;
            this_mae += all_mae[id];
            this_mse += all_mse[id];
        }
        // update statistics: skip the cluster with only one data point
        mae += this_mae; mse += this_mse;
    }
    mae /= (n*size); mse /= (n*size);
    
    delete[] all_mae;
    delete[] all_mse;
}

} // end namespace clustering
