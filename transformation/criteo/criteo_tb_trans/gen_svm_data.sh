#!/bin/bash

TRAIN=$1
TEST=$2
TH=$3

rm $TRAIN.feat
time ./txt2feat tr $TRAIN $TH
time ./feat2idpair.py $TRAIN.feat $TRAIN.id

rm $TEST.feat
time ./txt2feat te $TEST $TH
time ./feat2idpair.py --I $TEST.feat $TEST.id

rm $TRAIN.svm $TEST.svm
time ./txt2svm $TRAIN $TEST $TRAIN.id $TEST.id 8 --no_ffm
