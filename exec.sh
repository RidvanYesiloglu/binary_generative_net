#!/bin/bash
############################## MATLAB SCRIPT ######################################
#SBATCH --time=05:00:00
#SBATCH --job-name="ga_23_10"
#SBATCH --mail-user=ridvan@stanford.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=solution.OUT
#SBATCH --error=FAILURE.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#####################################

# Load MPI module (Enable MPI in user environment)
module load python/3.9.0
module load py-numpy/1.20.3_py39 
module load py-pytorch/1.8.1_py39
module load py-matplotlib/3.4.2_py39
module load cuda/11.2.0

# Change to the directory from which the batch job was submitted
export SLURM_SUBMIT_DIR=/home/users/ridvan/Low_Corr_Bin_Code_Design/
# Change to the job directory
cd $SLURM_SUBMIT_DIR
# nproc='cat $PBS_NODEFILE | wc -l'
# Run the MPI code

K=$1
N=$2
obj_no=$3
init_mode=$4
run_name=$5
wd_exists=$6
mc_samps=$7
eps=$8
no_runs=$9
python3 Train_Sigmoid_Model.py ${K} ${N} ${obj_no} ${init_mode} ${run_name} ${wd_exists} ${mc_samps} ${eps} ${no_runs}
