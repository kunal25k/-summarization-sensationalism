#!/bin/bash

memory=32
duration='48:00:00'
nodes=1
cpus=4
gres='gpu:rtx8000:1'

params=$1
model=$2
jobname=$3

file=out/${jobname}
error=${file}.err
out=${file}.out

sbatch --gres=$gres --mem=${memory}GB --time=$duration --job-name=$jobname --error=${error} --output=${out} --cpus-per-task=$cpus --nodes=$nodes ./jobs/summarization/eval/runner.sh $params $model
