#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <iostream>

int64_t feat_enc(int col, char *cur) {
  int64_t res = -1;
  if (col <= 13) { // I
    if (*cur) {
        int64_t x = atoll(cur);
        res = ((x + 1) << 7) + col;
    } else {
      res = col;
    }
  } else { // C
    if (*cur) {
      int64_t x = strtoll(cur, NULL, 16);
      res = ((x + 1) << 7) + col;
    } else {
      res = col;
    }
  }
  return res;
}

