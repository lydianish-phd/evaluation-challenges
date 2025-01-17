#!/bin/bash

module purge
module load pytorch-gpu/py3/2.3.0
source $HOME/.bashrc 
source $HOME/.bash_profile

set -e

GPT=gpt-4o-mini

# Non-standard English

CORPUS[0]=rocsmt
REF_FILE[0]=$DATASETS/rocsmt/test/raw.en.test
TGT_LANG[0]=French

CORPUS[1]=footweets
REF_FILE[1]=$DATASETS/footweets/detok.twitter.sent.de.txt
TGT_LANG[1]=German

# Non-standard French

CORPUS[2]=mmtc
REF_FILE[2]=$DATASETS/mmtc/test.fr-en.en
TGT_LANG[2]=English

CORPUS[3]=pfsmb
REF_FILE[3]=$DATASETS/pfsmb/test.en
TGT_LANG[3]=English


for i in {0..3}
do
    echo "Normalizing ${CORPUS[i]}..."
    python $HOME/evaluation-challenges/src/llm/normalize-gpt.py \
        --input-file ${REF_FILE[i]} \
        --target-lang ${TGT_LANG[i]} \
        --model-name $GPT 
done