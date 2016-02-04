#!/bin/sh

MODELS=$@

N=8
for m in $MODELS ; do ((i=i%N))
    ((i++==0)) && wait
    for s in `cat data/sites.txt` ; do
        scripts/run_model.py run $m $s
        for a in eval rst-gen ; do
            scripts/eval_model.py $a $m $s ;
        done
    done
    scripts/model_search_indexes.py model $m
done    

scripts/model_search_indexes.py all

