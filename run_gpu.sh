#!/bin/bash -l
#SBATCH --account=uludag_gpu
#SBATCH --job-name=test_gpu
#SBATCH --nodes=1
#SBATCH -p gpu #partition
#SBATCH --gres=gpu:1               # max 3 GPUs
#SBATCH --cpus-per-task=20         # max 41 CPUs
#SBATCH --mem=100G              # max > 200GB memory per node
#SBATCH --time=24:00:00 #wall time
#SBATCH --mail-user=brian.nghiem@rmp.uhn.ca
#SBATCH --mail-type=ALL #mail notifications

# Build output directory
export HOME=/cluster/projects/uludag/Brian/moco-sigpy/namer
cd ${HOME}

conda info --envs
'''Output:
base                  *  /cluster/tools/software/centos7/python/3.7.2
'''
conda activate /cluster/home/brian.nghiem/.conda/envs/sigpy_env
echo "now processing task id:: " ${SLURM_JOBID}
echo "--------------------------"

python ${HOME}/train_namer_cnn.py > ${HOME}/output_${SLURM_JOBID}.txt
