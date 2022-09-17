#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>

#include "criteo.hpp"

#define ERROR(x) perror(x);exit(0);

const int NUM_FIELD = 1+13+26;
const int BUF_SIZE = 333;

using std::string;
using std::unordered_map;
using std::ifstream;
using std::ofstream;
using std::to_string;

string data_type;
string data_path;
int threshold;

void process(char* s, unordered_map<string, int> &feat_cnt) {
  string tmps;
  char *cur = s;
  int l = (data_type == "te" ? 1: 0);
  int r = (data_type == "te" ? 14: NUM_FIELD);
  for (int i = l; i < r; ++i) {
    char *nxt = cur;
    for(; *nxt && *nxt != '\t' && *nxt != '\n'; ++nxt);
    *nxt = 0;
    if (i == 0) {
    } else {
      tmps = to_string(i) + " " + cur;
      if (i <= 13) {
        feat_cnt[tmps] = threshold + 1;
      } else {
        feat_cnt[tmps]++;
      }
    }
    cur = nxt+1;
  }
}

int main(int argc, char **argv) {
  static char buf[BUF_SIZE];
  if ( argc < 4 ) {
    ERROR("usage: ./txt2feat [type=tr or te] [training data] [threshold]");
  }
  data_type = argv[1];
  data_path = argv[2];
  threshold = atoi(argv[3]);

  FILE* fin = fopen(data_path.c_str(), "r");

  std::cerr << "threshold " << threshold << std::endl;

  int cnt = 0;
  unordered_map<string, int> feat_cnt;
  if (data_type == "tr") {
    std::cerr << "doing training data\n";
    for (int i = 14; i < NUM_FIELD; ++i) {
      string tmps = to_string(i) + " less";
      feat_cnt[tmps] = threshold + 1;
    }
  } else {
    std::cerr << "doing test data\n";
  }
  while (fgets(buf, BUF_SIZE, fin)) {
    process(buf, feat_cnt);
    ++cnt;
    if (cnt % 1000000 == 0)
      std::cerr << cnt << " processed\n";
  }
  fclose(fin);

  FILE* fo = fopen((data_path+".feat").c_str(), "a");

  for (auto p: feat_cnt) if (p.second >= threshold) {
    fprintf(fo, "%s %d\n", p.first.c_str(), p.second);
  }
  fclose(fo);
}


