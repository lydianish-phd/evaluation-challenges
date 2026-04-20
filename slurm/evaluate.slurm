#!/bin/bash

#SBATCH --job-name=evaluate      # Job name
#SBATCH --account=rnh@v100
#SBATCH --partition=gpu_p2l
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --gres=gpu:1        # numper of GPUs per node
#SBATCH --cpus-per-task=24	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=2:00:00                # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/evaluation-challenges/%x/%x_%j.log # Standard output and error log

echo "### Running $SLURM_JOB_NAME @ ###"

module purge
module load pytorch-gpu/py3/2.3.0
source $HOME/.bashrc 
source $HOME/.bash_profile

set -e

REPO_DIR=$HOME/evaluation-challenges
EXPERIMENT_DIR=$REPO_DIR/experiment_049d
DATA_DIR=$REPO_DIR/data_extracted
CORPORA_CONFIG=$REPO_DIR/src/config/corpora.yaml

TOWER=Unbabel/TowerInstruct-7B-v0.2
GEMMA=google/gemma-2-9b-it
LLAMA=meta-llama/Llama-3.1-8B-Instruct
NLLB=facebook/nllb-200-3.3B

ALL_MODELS=($NLLB $LLAMA $GEMMA $TOWER)
# CORPORA=(rocsmt footweets mmtc pfsmb rocsmt-gpt footweets-gpt mmtc-gpt pfsmb-gpt)
# GUIDELINES=(default standard general rocsmt footweets mmtc pfsmb)
CORPORA=(rocsmt footweets mmtc pfsmb)
GUIDELINES=(default rocsmt footweets mmtc pfsmb)

echo "Models: ${ALL_MODELS[@]}"
echo "Corpora: ${CORPORA[@]}"
echo "Guidelines: ${GUIDELINES[@]}"

python $HOME/evaluation-challenges/src/evaluate.py \
    --output-dir $EXPERIMENT_DIR/outputs \
    --data-dir $DATA_DIR \
    --corpora-config $CORPORA_CONFIG \
    --models ${ALL_MODELS[@]} \
    --corpora ${CORPORA[@]} \
    --guidelines ${GUIDELINES[@]}

python $HOME/evaluation-challenges/src/aggregate.py \
    --input-dir $EXPERIMENT_DIR \
    --models ${ALL_MODELS[@]} \
    --corpora ${CORPORA[@]} 

python $HOME/evaluation-challenges/src/analyze.py \
    --input-dir $EXPERIMENT_DIR 

echo "Done..."

