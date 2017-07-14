#!/bin/bash

set -e

if ! [ -d OpenBLAS-0.2.19 ]
then
    curl -L http://github.com/xianyi/OpenBLAS/archive/v0.2.19.tar.gz | tar -xz
    cd OpenBLAS-0.2.19
    make
    cd ..
fi

if command -v conda && ! [ -d cenv ]
then
    conda create -y -p cenv
    conda install -y -p cenv gensim==2.2.0
    python_commands='source activate cenv'
elif command -v virtualenv && ! [ -d venv ]
then
    virtualenv venv
    source venv/bin/activate
    pip install gensim==2.2.0
    deactivate
    python_commands='source venv/bin/activate'
else
    pip install --user -r requirements.txt
    python_commands=
fi

echo
echo '----------------------------------------------------------------------------------'
echo 'Use the following bash commands to add the installed software to your environment.'
echo
echo 'export LIBRARY_PATH="$PWD/OpenBLAS-0.2.19:$LIBRARY_PATH"'
echo 'export LD_LIBRARY_PATH="$PWD/OpenBLAS-0.2.19:$LD_LIBRARY_PATH"'
echo "$python_commands"
