#!/bin/bash

source $HOME/.bashrc 
source $HOME/.bash_profile

set -e

NLLB_EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_047k
LLM_EXPERIMENT_DIR=$EXPERIMENTS/robust-embeddings/sonar/experiment_049

CORPUS[0]=rocsmt
LANG_PAIR[0]=eng_Latn-fra_Latn

CORPUS[1]=footweets
LANG_PAIR[1]=eng_Latn-fra_Latn

CORPUS[2]=mmtc
LANG_PAIR[2]=fra_Latn-eng_Latn

CORPUS[3]=pfsmb
LANG_PAIR[3]=fra_Latn-eng_Latn

for i in {0..3}
do
    mkdir -p $LLM_EXPERIMENT_DIR/outputs/nllb3b/$CORPUS
    cp $NLLB_EXPERIMENT_DIR/outputs/nllb3b/${CORPUS[$i]}/${LANG_PAIR[$i]}/*.out $LLM_EXPERIMENT_DIR/outputs/facebook/nllb-200-3.3B/$CORPUS
done
