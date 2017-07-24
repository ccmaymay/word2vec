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

export C_INCLUDE_PATH="$OPENBLAS_PREFIX/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="$OPENBLAS_PREFIX/include:$CPLUS_INCLUDE_PATH"
export LIBRARY_PATH="$OPENBLAS_PREFIX/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$OPENBLAS_PREFIX/lib:$LD_LIBRARY_PATH"

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

if [ $# -eq 0 ]
then
    echo
    echo '--------------------------------------------------------------------------------'
    echo 'Use the following bash commands to add the installed software to your'
    echo 'environment.'
    echo '--------------------------------------------------------------------------------'
    echo
else
    echo
    echo '--------------------------------------------------------------------------------'
    echo 'Using the following bash commands to add the installed software to the'
    echo "environment, then doing: $@"
    echo '--------------------------------------------------------------------------------'
    echo
fi

echo 'export C_INCLUDE_PATH="$PWD/include:$C_INCLUDE_PATH"'
echo 'export CPLUS_INCLUDE_PATH="$PWD/include:$CPLUS_INCLUDE_PATH"'
echo 'export LIBRARY_PATH="$PWD/lib:$LIBRARY_PATH"'
echo 'export LD_LIBRARY_PATH="$PWD/lib:$LD_LIBRARY_PATH"'
echo "$python_commands"

if [ $# -gt 0 ]
then
    export C_INCLUDE_PATH="$PWD/include:$C_INCLUDE_PATH"
    export CPLUS_INCLUDE_PATH="$PWD/include:$CPLUS_INCLUDE_PATH"
    export LIBRARY_PATH="$PWD/lib:$LIBRARY_PATH"
    export LD_LIBRARY_PATH="$PWD/lib:$LD_LIBRARY_PATH"
    $python_commands
    "$@"
fi
