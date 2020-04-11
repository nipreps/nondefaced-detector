#!/bin/bash
#
#SBATCH -J split-training-e25           # Job name
#SBATCH -o split-training.o%j       # Name of stdout output file
#SBATCH -e split-training.e%j       # Name of stderr error file
#SBATCH -t 24:00:00        # Run time (hh:mm:ss)
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks
#SBATCH --mail-user=shashankbansal56@gmail.com

#SBATCH --mail-type=all    # Send email at begin and end of job


source ~/.bashrc
rm -r ~/.Logs
export CUDA_VISIBLE_DEVICES=0
python	-W ignore train.py -jn "split-training-epoch25"
