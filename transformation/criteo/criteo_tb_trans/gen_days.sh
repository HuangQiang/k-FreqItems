#!/usr/bin/env bash
touch tmp.txt
#for i in $(seq $1 $2); do
for i in $(seq 0 5); do
  echo $i
  # time ~/criteo-script/gen_svm_data.sh day_$i tmp.txt 50
  time bash gen_svm_data.sh /hdd1/1b/sparse/all/day_$i tmp.txt 50
done
