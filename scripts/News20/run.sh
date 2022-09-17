#!/bin/bash
rm filter
g++ -std=c++11 -w -O3 -o filter filter.cc 

ifile=geek.csv
ofile=geek_filter.csv
./filter ${ifile} ${ofile}
