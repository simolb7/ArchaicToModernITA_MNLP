#!/bin/bash
#SBATCH --job-name=finetuning         # Job name
#SBATCH --output=logs/log_prova_1.out          # Name of stdout output file. %x job_name, %j job_number
#SBATCH --error=logs/error_prova_1.err       # Name of stdout output file. %x job_name, %j job_number

#SBATCH -A try25_navigli                    # account name
#SBATCH -p boost_usr_prod              # quality of service
#SBATCH -t 24:00:00                    # time in HH:MM:SS for booster maximum 24H
#SBATCH -N 1
#SBATCH --ntasks=1                     # number of tasks, one should be enough
#SBATCH --ntasks-per-node=1           # number of tasks per node
#SBATCH --cpus-per-task=2             # number of CPU per tasks
#SBATCH --gres=gpu:1                  # number of GPU per node, 1 should be enough

# load the modules
module load profile/deeplrn cuda/12.1


# activate your environment
source $HOME/cleanEnv/bin/activate


export HF_HOME=/leonardo/home/userexternal/mkhan002/.cache/huggingface/hub
export HF_DATASETS_CACHE=/leonardo/home/userexternal/mkhan002/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=/leonardo/home/userexternal/mkhan002/.cache/huggingface/hub
export WANDB_MODE=offline # set the wandb offline (no needed for generation...)


# # read Huggingface token from .env file
export HF_TOKEN=$(python -c "import huggingface_hub; print(huggingface_hub.HfFolder.get_token() or '')")


# # execute the python script
python finetuning_llama.py
