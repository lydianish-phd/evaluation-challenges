#!/bin/bash

source $HOME/.bashrc 
source $HOME/.bash_profile

set -e

NLLB_EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047k
LLM_EXPERIMENT_DIR=$EXPERIMENTS/evaluation-challenges/llm/experiment_049

CORPUS[0]=rocsmt
LANG_PAIR[0]=eng_Latn-fra_Latn

CORPUS[1]=footweets
LANG_PAIR[1]=eng_Latn-deu_Latn

CORPUS[2]=mmtc
LANG_PAIR[2]=fra_Latn-eng_Latn

CORPUS[3]=pfsmb
LANG_PAIR[3]=fra_Latn-eng_Latn

for i in {0..3}
do
    OUTPUT_DIR=$LLM_EXPERIMENT_DIR/facebook/nllb-200-3.3B/$CORPUS
    mkdir -p $OUTPUT_DIR
    cp -v $NLLB_EXPERIMENT_DIR/outputs/nllb3b/${CORPUS[$i]}/${LANG_PAIR[$i]}/*.out $OUTPUT_DIR
done

echo "Done copying NLLB outputs to LLM experiment directory..."
