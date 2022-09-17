#!/bin/bash

# ------------------------------------------------------------------------------
#  convert criteo day data into sparse format
# ------------------------------------------------------------------------------
g++ -std=c++11 -w -O3 -o criteo criteo.cc

d=1000000

n=195841983
input="day_0.svm"
output="Criteo_Day0_0.bin"
./criteo ${n} ${d} ${input} ${output}

n=199563535
input="day_1.svm"
output="Criteo_Day1_0.bin"
./criteo ${n} ${d} ${input} ${output}

n=196792019
input="day_2.svm"
output="Criteo_Day2_0.bin"
./criteo ${n} ${d} ${input} ${output}

n=181115208
input="day_3.svm"
output="Criteo_Day3_0.bin"
./criteo ${n} ${d} ${input} ${output}

n=152115810
input="day_4.svm"
output="Criteo_Day4_0.bin"
./criteo ${n} ${d} ${input} ${output}

n=172548507
input="day_5.svm"
output="Criteo_Day5_0.bin"
./criteo ${n} ${d} ${input} ${output}

# ------------------------------------------------------------------------------
#  extract 1 billion & 10 million data from day_0 ~ day_5
# ------------------------------------------------------------------------------
g++ -std=c++11 -w -O3 -o criteo_1b criteo_1b.cc
./criteo_1b
