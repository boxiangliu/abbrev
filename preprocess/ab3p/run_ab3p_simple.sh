#!/bin/bash

in_fn=$1
# in_fn=../data/BioC/Ab3P-BioC/Ab3P_bioc_gold.txt 
identify_abbr=/mnt/scratch/boxiang/projects/Ab3P/identify_abbr
echo "/mnt/scratch/boxiang/projects/Ab3P/WordData/" > path_Ab3P

$identify_abbr $in_fn

rm path_Ab3P