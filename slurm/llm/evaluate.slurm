#!/bin/bash

#SBATCH --job-name=evaluate      # Job name
#SBATCH --account=rnh@v100
#SBATCH --partition=gpu_p2l
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --gres=gpu:1        # numper of GPUs per node
#SBATCH --cpus-per-task=24	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=4:00:00                # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/rnh/udc54vm/logs/evaluation-challenges/llm/%x/%x_%j.log # Standard output and error log

echo "### Running $SLURM_JOB_NAME @ ###"

module purge
module load pytorch-gpu/py3/2.3.0
source $HOME/.bashrc 
source $HOME/.bash_profile

set -e

EXPERIMENT_DIR=$EXPERIMENTS/evaluation-challenges/llm/experiment_049
CORPORA_CONFIG=$HOME/evaluation-challenges/src/llm/config/corpora.yaml

LLAMA=meta-llama/Llama-3.1-8B-Instruct
GEMMA=google/gemma-2-9b-it
NLLB=facebook/nllb-200-3.3B

MODELS=($LLAMA $GEMMA $NLLB)
CORPORA=(rocsmt footweets mmtc pfsmb)
GUIDELINES=(default standard general)

echo "Models: ${MODELS[@]}"
echo "Corpora: ${CORPORA[@]}"
echo "Guidelines: ${GUIDELINES[@]}"

python $HOME/evaluation-challenges/src/llm/evaluate.py \
    --output-dir $EXPERIMENT_DIR/outputs \
    --corpora-config $CORPORA_CONFIG \
    --models ${MODELS[@]} \
    --corpora ${CORPORA[@]} \
    --guidelines ${GUIDELINES[@]}

python $HOME/evaluation-challenges/src/llm/aggregate.py \
    --input-dir $EXPERIMENT_DIR \
    --models ${MODELS[@]} \
    --corpora ${CORPORA[@]} \

echo "Done..."

