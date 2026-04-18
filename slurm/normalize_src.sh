#!/bin/bash

INDEX=$1
if [ -z $INDEX ]; then
    INDEX=0
fi

GPT=gpt-4o-mini

# Non-standard English

CORPUS[0]=rocsmt
REF_FILE[0]=$DATASETS/rocsmt/test/raw.en.test
SRC_LANG[0]=English

CORPUS[1]=footweets
REF_FILE[1]=$DATASETS/footweets/detok.twitter.sent.en.txt
SRC_LANG[1]=English

CORPUS[2]=mtnt
REF_FILE[2]=$DATASETS/mtnt/MTNT2019/en-fr.en
SRC_LANG[2]=English

CORPUS[3]=mtnt
REF_FILE[3]=$DATASETS/mtnt/MTNT2019/en-ja.en
SRC_LANG[3]=English

# Non-standard French

CORPUS[4]=foursquare
REF_FILE[4]=$DATASETS/foursquare/test.fr
SRC_LANG[4]=French

CORPUS[5]=mtnt
REF_FILE[5]=$DATASETS/mtnt/MTNT2019/fr-en.fr
SRC_LANG[5]=French

CORPUS[6]=mmtc
REF_FILE[6]=$DATASETS/mmtc/test.fr-en.fr
SRC_LANG[6]=French

CORPUS[7]=pfsmb
REF_FILE[7]=$DATASETS/pfsmb/test.fr
SRC_LANG[7]=French


echo "Normalizing ${CORPUS[$INDEX]}..."
python $HOME/evaluation-challenges/src/normalize-gpt.py \
    --input-file ${REF_FILE[$INDEX]} \
    --target-lang ${SRC_LANG[$INDEX]} \
    --model-name $GPT 
