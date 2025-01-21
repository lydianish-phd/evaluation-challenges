#!/bin/bash

#SBATCH --job-name=generate      # Job name
#SBATCH --account=rnh@a100
#SBATCH --partition=gpu_p5
#SBATCH -C a100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --gres=gpu:1        # numper of GPUs per node
#SBATCH --cpus-per-task=64	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=0:20:00                # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/evaluation-challenges/llm/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0-3

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
# Using pytorch-gpu/py3/2.5.0 which has pytorch 2.4.0 and vllm 0.6.1
module load arch/a100 pytorch-gpu/py3/2.5.0
source $HOME/.bashrc 
source $HOME/.bash_profile

set -e

EXPERIMENT_DIR=$EXPERIMENTS/evaluation-challenges/llm/experiment_049

MODEL_NAME[0]=meta-llama/Llama-3.1-8B-Instruct
MODEL_DIR[0]=$MODELS/${MODEL_NAME[0]}

MODEL_NAME[1]=google/gemma-2-9b-it
MODEL_DIR[1]=$MODELS/${MODEL_NAME[1]}

# Non-standard English

CORPUS[0]=rocsmt
DATA_DIR[0]=$DATASETS/rocsmt/test
INPUT_FILE[0]=${DATA_DIR[0]}/raw.en.test
TGT_LANG[0]=French

CORPUS[1]=footweets
DATA_DIR[1]=$DATASETS/footweets
INPUT_FILE[1]=${DATA_DIR[1]}/detok.twitter.sent.en.txt
TGT_LANG[1]=German

# Non-standard French

CORPUS[2]=mmtc
DATA_DIR[2]=$DATASETS/mmtc
INPUT_FILE[2]=${DATA_DIR[2]}/test.fr-en.fr
TGT_LANG[2]=English

CORPUS[3]=pfsmb
DATA_DIR[3]=$DATASETS/pfsmb
INPUT_FILE[3]=${DATA_DIR[3]}/test.fr
TGT_LANG[3]=English

GUIDELINES=(default standard general rocsmt footweets mmtc pfsmb)
CONFIG=$HOME/evaluation-challenges/src/llm/config/greedy.yaml

for i in {0..1}
do
    echo "Model: ${MODEL_NAME[$i]}"

    OUTPUT_DIR=$EXPERIMENT_DIR/outputs/${MODEL_NAME[$i]}/${CORPUS[$SLURM_ARRAY_TASK_ID]}    

    python $HOME/evaluation-challenges/src/llm/generate.py \
        -i ${INPUT_FILE[$SLURM_ARRAY_TASK_ID]} \
        -o $OUTPUT_DIR \
        -m ${MODEL_DIR[$i]} \
        -c $CONFIG \
        -l ${TGT_LANG[$SLURM_ARRAY_TASK_ID]} \
        -g ${GUIDELINES[@]} 

done

echo "Done..."

