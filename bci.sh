#!/usr/bin/env bash

rm -rf data
mkdir -p data/BCI

for((i=1; i<=9; i++))

do

python generate_data.py --subject_id $i

done

#python conformer.py

for((i=1; i<=9; i++))

do

python convnet.py --subject_id $i

done