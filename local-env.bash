#!/bin/bash

set -e

if ! [ -d OpenBLAS-0.2.19 ]
then
    curl -L http://github.com/xianyi/OpenBLAS/archive/v0.2.19.tar.gz | tar -xz
    cd OpenBLAS-0.2.19
    make
    cd ..
fi

export C_INCLUDE_PATH="$PWD/OpenBLAS-0.2.19:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="$PWD/OpenBLAS-0.2.19:$CPLUS_INCLUDE_PATH"
export LIBRARY_PATH="$PWD/OpenBLAS-0.2.19:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$PWD/OpenBLAS-0.2.19:$LD_LIBRARY_PATH"
export BLAS="$PWD/OpenBLAS-0.2.19/libopenblas.so"
export LAPACK=

if command -v conda
then
    if ! [ -d cenv ]
    then
        conda create -y -p cenv
        conda install -y -p cenv numpy
        conda install -y -p cenv gensim
    fi
    python_commands='source activate cenv'
elif command -v virtualenv
then
    if ! [ -d venv ]
    then
        virtualenv venv
        source venv/bin/activate
        pip install numpy
        pip install gensim
        deactivate
    fi
    python_commands='source venv/bin/activate'
else
    pip install --user numpy
    pip install --user gensim
    python_commands=
fi

echo
echo '----------------------------------------------------------------------------------'
echo 'Use the following bash commands to add the installed software to your environment.'
echo
echo 'export C_INCLUDE_PATH="$PWD/OpenBLAS-0.2.19:$C_INCLUDE_PATH"'
echo 'export CPLUS_INCLUDE_PATH="$PWD/OpenBLAS-0.2.19:$CPLUS_INCLUDE_PATH"'
echo 'export LIBRARY_PATH="$PWD/OpenBLAS-0.2.19:$LIBRARY_PATH"'
echo 'export LD_LIBRARY_PATH="$PWD/OpenBLAS-0.2.19:$LD_LIBRARY_PATH"'
echo "$python_commands"
