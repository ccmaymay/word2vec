#!/bin/bash

set -e

if ! [ -d lib ]
then
    curl -L http://github.com/xianyi/OpenBLAS/archive/v0.2.19.tar.gz | tar -xz
    OPENBLAS_PREFIX=$PWD
    cd OpenBLAS-0.2.19
    make
    make install PREFIX=$OPENBLAS_PREFIX
    cd ..
    rm -rf OpenBLAS-0.2.19
fi

export C_INCLUDE_PATH="$OPENBLAS_PREFIX:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="$OPENBLAS_PREFIX:$CPLUS_INCLUDE_PATH"
export LIBRARY_PATH="$OPENBLAS_PREFIX:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$OPENBLAS_PREFIX:$LD_LIBRARY_PATH"

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
        pip install --upgrade pip
        pip install numpy
        pip install gensim
        deactivate
    fi
    python_commands='source venv/bin/activate'
else
    echo >&2
    echo 'Error: please install conda or virtualenv before running this script.' >&2
    exit 1
fi

echo
echo '----------------------------------------------------------------------------------'
echo 'Use the following bash commands to add the installed software to your environment.'
echo
echo 'export C_INCLUDE_PATH="$PWD:$C_INCLUDE_PATH"'
echo 'export CPLUS_INCLUDE_PATH="$PWD:$CPLUS_INCLUDE_PATH"'
echo 'export LIBRARY_PATH="$PWD:$LIBRARY_PATH"'
echo 'export LD_LIBRARY_PATH="$PWD:$LD_LIBRARY_PATH"'
echo "$python_commands"
