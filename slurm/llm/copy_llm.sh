#!/bin/bash

source $HOME/.bashrc 
source $HOME/.bash_profile

set -e

EXPERIMENT_DIR=$EXPERIMENTS/evaluation-challenges/llm/experiment_049

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


for i in {0..2}
do
    echo "Model: ${MODEL_NAME[$i]}"

    for j in {0..3}
    do
        echo " - Corpus: ${CORPUS[$j]}"
        
        OUTPUT_DIR=$EXPERIMENT_DIR/outputs/${MODEL_NAME[$i]}/${CORPUS[$j]}
        
        mkdir -p ${OUTPUT_DIR}-gpt
        cp -v $OUTPUT_DIR/*.out ${OUTPUT_DIR}-gpt
    done
done

echo "Done copying NLLB baseline and default LLM outputs..."
