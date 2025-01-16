#!/bin/bash

#SBATCH --job-name=evaluate      # Job name
#SBATCH --account=rnh@v100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --gres=gpu:1        # numper of GPUs per node
#SBATCH --cpus-per-task=40	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=0:30:00                # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/evaluation-challenges/llm/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0-3

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
module load arch/a100 pytorch-gpu/py3/2.3.0
source $HOME/.bashrc 
source $HOME/.bash_profile

set -e

EXPERIMENT_DIR=$EXPERIMENTS/evaluation-challenges/llm/experiment_049
MODEL_NAME[0]=meta-llama/Llama-3.1-8B-Instruct
MODEL_NAME[1]=google/gemma-2-9b-it
MODEL_NAME[2]=facebook/nllb-200-3.3B

# Non-standard English

CORPUS[0]=rocsmt
DATA_DIR[0]=$DATASETS/${CORPUS[0]}/test
SRC_FILE_NAME[0]=raw.en.test
SRC_FILE[0]=${DATA_DIR[0]}/${SRC_FILE_NAME[0]}
REF_FILE[0]=${DATA_DIR[0]}/ref.fr.test

CORPUS[1]=footweets
DATA_DIR[1]=$DATASETS/${CORPUS[1]}
SRC_FILE_NAME[1]=detok.twitter.sent.en.txt
SRC_FILE[1]=${DATA_DIR[1]}/${SRC_FILE_NAME[1]}
REF_FILE[1]=${DATA_DIR[1]}/detok.twitter.sent.de.txt

# Non-standard French

CORPUS[2]=mmtc
DATA_DIR[2]=$DATASETS/${CORPUS[2]}
SRC_FILE_NAME[2]=test.fr-en.fr
SRC_FILE[2]=${DATA_DIR[2]}/${SRC_FILE_NAME[2]}
REF_FILE[2]=${DATA_DIR[2]}/test.fr-en.en

CORPUS[3]=pfsmb
DATA_DIR[3]=$DATASETS/${CORPUS[3]}
SRC_FILE_NAME[3]=test.fr
SRC_FILE[3]=${DATA_DIR[3]}/${SRC_FILE_NAME[3]}
REF_FILE[3]=${DATA_DIR[3]}/test.en

GUIDELINES=(default standard general)
SYS_FILES=()

echo "Models: ${MODEL_NAME[@]}"

for MODEL in ${MODEL_NAME[@]}
do
    OUTPUT_DIR=$EXPERIMENT_DIR/outputs/$MODEL/${CORPUS[$SLURM_ARRAY_TASK_ID]}
    for GUIDELINE in ${GUIDELINES[@]}
    do
        SYS_FILE_SUFFIX=${SRC_FILE_NAME[$SLURM_ARRAY_TASK_ID]}.$GUIDELINE.out
        SYS_FILES+=("$OUTPUT_DIR/$SYS_FILE_SUFFIX")
    done
done

python $HOME/evaluation-challenges/src/llm/evaluate.py \
    --src-file ${SRC_FILE[$SLURM_ARRAY_TASK_ID]} \
    --sys-files "${SYS_FILES[@]}" \
    --ref-file ${REF_FILE[$SLURM_ARRAY_TASK_ID]}

echo "Done..."

