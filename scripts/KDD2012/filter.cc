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

#include <unistd.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>

// -----------------------------------------------------------------------------
struct Result {
    int   k_;
    float time_;
    float d2c_;
    float radius_;
    int   k2_;
    float mae_;
    float mse_;
    std::string others_;
};

// -----------------------------------------------------------------------------
struct less_than_key {
    inline bool operator() (const Result& r1, const Result& r2) {
        if (r1.k_ < r2.k_) {
            return true;
        } 
        else if (r1.k_ > r2.k_) {
            return false;
        } 
        else {
            if (r1.time_ < r2.time_) return true;
            else return false;
        }
    }
};

// -----------------------------------------------------------------------------
void create_dir(                    // create dir if the path does not exist
    char *path)                         // input path
{
    int len = (int) strlen(path);
    for (int i = 0; i < len; ++i) {
        if (path[i] == '/') {
            char ch = path[i+1]; path[i+1] = '\0';
            
            if (access(path, F_OK) != 0) {
                if (mkdir(path, 0755) != 0) {
                    printf("Could not create %s\n", path); exit(1);
                }
            }
            path[i+1] = ch;
        }
    }
}

// -----------------------------------------------------------------------------
void get_csv_from_line(             // get an array with csv format from a line
    const std::string &str_data,        // a string line
    Result &csv_data)                   // csv data (return)
{
    std::istringstream ss(str_data);
    std::string tmp;
    
    getline(ss, tmp, ','); csv_data.k_      = stoi(tmp);
    getline(ss, tmp, ','); csv_data.time_   = stof(tmp);
    getline(ss, tmp, ','); csv_data.d2c_    = stof(tmp);
    getline(ss, tmp, ','); csv_data.radius_ = stof(tmp);
    getline(ss, tmp, ','); csv_data.k2_     = stoi(tmp);
    getline(ss, tmp, ','); csv_data.mae_    = stof(tmp);
    getline(ss, tmp, ','); csv_data.mse_    = stof(tmp);

    getline(ss, tmp); csv_data.others_ = tmp;
}

// -----------------------------------------------------------------------------
void read_csv(                      // read csv data
    const char *fname,                  // file name of query ids set
    std::string &title,                 // title (return)
    std::vector<Result> &results)       // results (return)
{
    std::ifstream infile(fname);
    if (!infile) { printf("Could not open %s\n", fname); exit(1); }

    // read title
    getline(infile, title);

    // read results
    std::string tmp;
    while (infile) {
        getline(infile, tmp);
        if (tmp.size() > 0 && tmp[0] >= '0' && tmp[0] <= '9') {
            Result result;
            get_csv_from_line(tmp, result);
            results.push_back(result);
        }
    }
    infile.close();
}

// -----------------------------------------------------------------------------
void filter(                        // filter results by d2c and radius
    float diff,                         // allow difference
    float err,                          // allow error
    const std::vector<Result> &results, // original results
    std::vector<Result> &filter_results)// filter results
{
    Result old = results[0];
    filter_results.push_back(old);

    for (int i = 1; i < (int) results.size(); ++i) {
        Result cur = results[i];
        filter_results.push_back(cur);
        
        // if (cur.mse_ < old.mse_ || (cur.mse_ - old.mse_ < 0.001 && cur.time_ < old.time_)) {
        //     filter_results.push_back(cur);
        //     old = cur;
        // }
        
        // if (cur.d2c_ < old.d2c_ && cur.radius_ < old.radius_) {
        //     filter_results.push_back(cur);
        //     old = cur;
        // }
        // if (old.d2c_ - cur.d2c_ > diff && fabs(cur.radius_ - old.radius_) < err) {
        //     filter_results.push_back(cur);
        //     old = cur;
        // }
        // else if (old.radius_ - cur.radius_ > diff && fabs(cur.d2c_ - old.d2c_) < err) {
        //     filter_results.push_back(cur);
        //     old = cur;
        // }
        // else if (cur.d2c_ < old.d2c_ && cur.time_ < old.time_) {
        //     filter_results.push_back(cur);
        //     old = cur;
        // }
        // else if (cur.radius_ < old.radius_ && cur.time_ < old.time_) {
        //     filter_results.push_back(cur);
        //     old = cur;
        // }
    }
}

// -----------------------------------------------------------------------------
void write_results(                 // write filter results
    const char *fname,                  // file name
    const std::string title,            // title
    const std::vector<Result> &results) // filter results
{
    FILE *fp = fopen(fname, "w");
    if (!fp) { printf("Could not open %s\n", fname); exit(1); }

    fprintf(fp, "%s\n", title.data());
    for (auto &res : results) {
        fprintf(fp, "%d,%.2f,%.4f,%.4f,%d,%.4f,%.4f,%s\n", res.k_, res.time_, 
            res.d2c_, res.radius_, res.k2_, res.mae_, res.mse_, res.others_.data());
    }
    fclose(fp);
}

// -----------------------------------------------------------------------------
int main(int argc, char **argv) 
{
    char ifname[200]; strcpy(ifname, argv[1]);
    char ofname[200]; strcpy(ofname, argv[2]);
    create_dir(ofname);

    // read csv data and get title and results
    std::string title;
    std::vector<Result> results;
    read_csv(ifname, title, results);

    // sort results in ascending order of num_clusters and time
    std::sort(results.begin(), results.end(), less_than_key());

    // filter results by d2c and radius
    float diff = 0.003f;
    float err  = 0.001f;
    std::vector<Result> filter_results;
    filter(diff, err, results, filter_results);

    // write filter results to disk
    write_results(ofname, title, filter_results);

    return 0;
}
