#!/bin/bash
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=2:00:00,num_proc=1,mem_free=4G

set -e

source ~/.bashrc

hostname -s

PREFIX=$HOME/`hostname -s`
echo PREFIX $PREFIX

cd
mkdir -p $PREFIX
cd $PREFIX
if [ ! -d word2vec ]
then
    git clone -b runtime-analysis ~/word2vec
fi
cd word2vec

echo LIBRARY_PATH $LIBRARY_PATH
echo LD_LIBRARY_PATH $LD_LIBRARY_PATH
rm -f word2vec-blas
CBLAS_FLAGS=-lcblas NUM_TRIALS=3 make runtime-word2vec-blas.tab
ldd word2vec-blas
cat runtime-word2vec-blas.tab

tar -xzf ~/OpenBLAS-0.2.19.tar.gz
cd OpenBLAS-0.2.19
make && make install PREFIX=$PREFIX
cd ..

export LIBRARY_PATH=$PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH

echo LIBRARY_PATH $LIBRARY_PATH
echo LD_LIBRARY_PATH $LD_LIBRARY_PATH
rm -f word2vec-blas
CBLAS_FLAGS=-lopenblas NUM_TRIALS=3 make runtime-word2vec-blas.tab
ldd word2vec-blas
cat runtime-word2vec-blas.tab
