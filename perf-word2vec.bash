#!/bin/bash

set -e

for e in dTLB-loads,dTLB-load-misses iTLB-loads,iTLB-load-misses L1-icache-loads,L1-icache-load-misses cycles,instructions L1-dcache-loads,L1-dcache-load-misses LLC-loads,LLC-load-misses L1-dcache-prefetches
do
    for p in word2vec-true-1-neg word2vec-false-1-neg word2vec-alias-neg word2vec-unsmoothed-neg word2vec word2vec-uniform-neg
    do
        for i in {1..10}
        do
            echo $p
            perf stat -e $e ./$p -train text8 -read-vocab vocab -output /dev/null -cbow 0 -hs 0 -binary 1 -iter 1 -threads 1 2>&1 1>/dev/null
        done
    done
done
