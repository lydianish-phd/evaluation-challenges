#!/bin/bash

source $HOME/.bashrc 
source $HOME/.bash_profile

set -e

INPUT_DIR=$EXPERIMENTS/evaluation-challenges/llm/experiment_049
OUTPUT_DIR=$EXPERIMENTS/evaluation-challenges/llm/experiment_049b

# Models

MODEL_NAME[0]=meta-llama/Llama-3.1-8B-Instruct
MODEL_NAME[1]=google/gemma-2-9b-it
MODEL_NAME[2]=facebook/nllb-200-3.3B

# Corpora

CORPUS[0]=rocsmt
CORPUS[1]=footweets
CORPUS[2]=mmtc
CORPUS[3]=pfsmb
CORPUS[4]=rocsmt-gpt
CORPUS[5]=footweets-gpt
CORPUS[6]=mmtc-gpt
CORPUS[7]=pfsmb-gpt

echo "Copying baseline and default outputs from $INPUT_DIR to $OUTPUT_DIR"

for i in {0..2}
do
    echo "Model: ${MODEL_NAME[$i]}"

    for j in {0..7}
    do
        echo " - Corpus: ${CORPUS[$j]}"
        
        mkdir -p $OUTPUT_DIR/outputs/${MODEL_NAME[$i]}/${CORPUS[$j]}
        if [ $i -eq 2 ]; then # NLLB
            cp -v $INPUT_DIR/outputs/${MODEL_NAME[$i]}/${CORPUS[$j]}/*.out $OUTPUT_DIR/outputs/${MODEL_NAME[$i]}/${CORPUS[$j]}
        else
            cp -v $INPUT_DIR/outputs/${MODEL_NAME[$i]}/${CORPUS[$j]}/*default.out $OUTPUT_DIR/outputs/${MODEL_NAME[$i]}/${CORPUS[$j]}
        fi
        
    done
done
