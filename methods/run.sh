#!/bin/bash

# scl enable devtoolset-4 -- bash

./run_url_kfreqitems.sh 
./run_criteo10m_kfreqitems.sh 
./run_avazu_kfreqitems.sh 
./run_kdd2012_kfreqitems.sh 
./run_criteo1b_kfreqitems.sh 

./run_url_silk.sh 
./run_criteo10m_silk_4.sh 
./run_avazu_silk_4.sh 
./run_kdd2012_silk.sh 
./run_criteo1b_silk.sh 

./run_criteo10m_silk_1.sh 
./run_criteo10m_silk_2.sh 
./run_avazu_silk_1.sh 
./run_avazu_silk_2.sh

./run_criteo10m_kfreqitems.sh 
./run_avazu_kfreqitems.sh 
./run_criteo10m_silk_8.sh 
./run_avazu_silk_8.sh
