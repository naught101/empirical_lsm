#!/bin/bash

set -ex

MODELS=$*

N=3
i=0
for m in $MODELS ; do
    i=$(echo "$i%$N" | bc)
    i=$(echo "$i+1" | bc)
    if [[ $i -eq 0 ]] ; then wait ; fi
    (
    for s in `cat data/PALS/datasets/sites_PLUMBER_ext.txt` ; do
        scripts/offline_runs/run_model.py run $m $s
        scripts/offline_runs/eval_model.py eval $m $s ;
        scripts/offline_runs/eval_model.py rst-gen $m $s ;
    done
    )
    scripts/offline_runs/model_search_indexes.py model $m &
done    

wait

scripts/offline_runs/model_search_indexes.py summary

make html
