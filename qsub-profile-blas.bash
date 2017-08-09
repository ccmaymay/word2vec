#!/bin/bash

if [ $# -ne 1 ]
then
    echo "usage: $0 <grid>" >&2
    exit 1
fi

if [ "$1" == clsp ]
then
    qhost | egrep '[abghy][0-9][0-9]\.clsp\.jhu\.edu' | awk '{ print $1 }' | while read host
    do
        qsub -q all.q@$host profile-blas.bash
    done
elif [ "$1" == coe ]
then
    qhost | egrep 'r[78]n[0-9]+' | grep -v r8n5 | awk '{ print $1 }' | while read host
    do
        qsub -q all.q@$host profile-blas.bash
    done
else
    echo "unknown grid $1" >&2
    exit 1
fi

