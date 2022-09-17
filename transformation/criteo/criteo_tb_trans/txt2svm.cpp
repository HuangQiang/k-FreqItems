#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "criteo.hpp"

#define ERROR(x) perror(x);exit(0);

const int NUM_FIELD = 1+13+26;
const int BATCH_SIZE = 5e7;
const int BUF_SIZE  = 333;

using std::string;
using std::unordered_map;
using std::unordered_set;
using std::ifstream;
using std::ofstream;
using std::to_string;
using std::vector;

bool flag_svm;
bool flag_ffm;

string train_path;
string test_path;
string feat_path;
string itype_path;

int threshold;
int n_bins;
int n_thread;
bool* freq_feat;

int n_train;
char **data;
vector<int> *feat;

unordered_map<int64_t, int> feat2id;

int hash_feat(int col, int64_t x) {
  if (feat2id.count(x)) {
    return feat2id[x];
  } else {
    int64_t col_less = col + (1LL << 60);
    return feat2id[col_less];
  }
}

void read_feature_to_id() {
  int64_t x;
  int id;
  FILE* fi = fopen(feat_path.c_str(), "r");
  while(fscanf(fi, "%lld%d", &x, &id) != EOF) {
    feat2id[x] = id;
  }
  fi = fopen(itype_path.c_str(), "r");
  while(fscanf(fi, "%lld%d", &x, &id) != EOF) {
    feat2id[x] = id;
  }
}

void process(char *s, vector<int> &feat_ids) {
  char *cur = s;
  feat_ids.resize(NUM_FIELD-1);
  for (int i = 0; i < NUM_FIELD; ++i) {
    char *nxt = cur;
    for(; *nxt && *nxt != '\t' && *nxt != '\n'; ++nxt);
    *nxt = 0;
    if (i != 0) {
      int64_t the_feat = feat_enc(i, cur);
      feat_ids[i - 1] = hash_feat(i, the_feat);
    }
    cur = nxt+1;
  }
}

void work_train() {
  FILE* fin = fopen(train_path.c_str(), "r");
  FILE* fosvm = fopen((train_path+".svm").c_str(), "a");
  FILE* foffm = fopen((train_path+".ffm").c_str(), "a");

  for (int n_batch = 0; ; ++n_batch) {
    std::cerr << "batch " << n_batch << '\n';
    n_train = BATCH_SIZE;
    for (int i = 0; i < n_train; ++i) {
      if (!fgets(data[i], BUF_SIZE-1, fin)) {
        n_train = i;
        break;
      }
    }
    if (n_train == 0) break;

    std::cerr << "training data input done\n";
    std::cerr << "OPENMP\n";
    int cnt = 0;
#pragma omp parallel for firstprivate(cnt)
    for (int i = 0; i < n_train; ++i) {
      process(data[i], feat[i]);
      ++cnt;
      if (cnt % 1000000 == 0) {
        std::cerr << cnt << "  processed\n";
      }
    }

    std::cerr << "train 2 feat done\n";


    cnt = 0;
    for (int i = 0; i < n_train; ++i) {
      vector<int> &feat_ids = feat[i];
      if (flag_ffm) {
        fprintf(foffm, "%c", data[i][0]);
        for (size_t j = 0; j < feat_ids.size(); ++j) {
          fprintf(foffm, " %d:%d:1", (int)j, feat_ids[j]);
        }
        fprintf(foffm, "\n");
      }
      if (flag_svm) {
        sort(feat_ids.begin(), feat_ids.end());
        feat_ids.resize(unique(feat_ids.begin(), feat_ids.end()) - feat_ids.begin());
        double val = 1. / sqrt(feat_ids.size());
        fprintf(fosvm, "%c", data[i][0]);
        for (int id: feat_ids) fprintf(fosvm, " %d:%.5f", id, val);
        fprintf(fosvm, "\n");
      }
      ++cnt;
      if (cnt % 1000000 == 0)
        std::cerr << cnt << " output\n";
    }
    fflush(fosvm);
  }

}

void work_test() {
  static char buf[BUF_SIZE];
  static char w_label[BUF_SIZE];
  FILE* fin = fopen(test_path.c_str(), "r");
  FILE* fosvm = fopen((test_path+".svm").c_str(), "a");
  FILE* foffm = fopen((test_path+".ffm").c_str(), "a");

  int cnt = 0;
  vector<int> feat_ids;
  while (fgets(buf, BUF_SIZE, fin)) {
    snprintf(w_label, BUF_SIZE, "0\t%s", buf);
    process(w_label, feat_ids);

    if (flag_ffm) {
      fprintf(foffm, "0");
      for (size_t j = 0; j < feat_ids.size(); ++j) {
        fprintf(foffm, " %d:%d:1", (int)j, feat_ids[j]);
      }
      fprintf(foffm, "\n");
    }
    if (flag_svm) {
      sort(feat_ids.begin(), feat_ids.end());
      feat_ids.resize(unique(feat_ids.begin(), feat_ids.end()) - feat_ids.begin());
      double val = 1. / sqrt(feat_ids.size());
      fprintf(fosvm, "0");
      for (int id: feat_ids) fprintf(fosvm, " %d:%.5f", id, val);
      fprintf(fosvm, "\n");
    }
    feat_ids.clear();

    ++cnt;
    if (cnt % 1000000 == 0)
      std::cerr << cnt << " processed\n";
  }

}

int main(int argc, char **argv) {
  if (argc < 5) {
    ERROR("usage: ./txt2svm [training data] [test data] [feature to id data] [I type of test to id data] [# of thread] [--no_svm if don't output as LIBSVM format] [--no_ffm if don't output as LIBFFM format]");
  }
  flag_svm = flag_ffm = 1;
  train_path = argv[1];
  test_path  = argv[2];
  feat_path  = argv[3];
  itype_path = argv[4];
  n_thread   = atoi(argv[5]);
  for (int i = 6; i < argc; ++i) {
    if (strcmp(argv[i], "--no_svm") == 0) {
      flag_svm = 0;
    } else if (strcmp(argv[i], "--no_ffm") == 0) {
      flag_ffm = 0;
    }
  }
#ifdef _OPENMP
  omp_set_num_threads(n_thread);
#endif
  data = new char*[BATCH_SIZE];
  feat = new vector<int>[BATCH_SIZE];
  for (int i = 0; i < BATCH_SIZE; ++i) {
    data[i] = new char[BUF_SIZE];
    feat[i] = vector<int>();
  }

  read_feature_to_id();

  work_train();
  work_test();

}

