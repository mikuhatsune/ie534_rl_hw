#!/bin/bash
#PBS -l nodes=1:ppn=9:xk
#PBS -l walltime=14:00:00
#PBS -N breakout_a2c
#PBS -e $PBS_JOBID.err
#PBS -o $PBS_JOBID.out
# -m and -M set up mail messages at begin,end,abort:
# -m bea
# -M YOUR_NETID@illinois.edu

#. /opt/modules/default/init/bash
module load python/2.0.0
pip install gym[atari]==0.14 --user
#module load cudatoolkit
aprun -n 1 -N 1 python Main.py --algo a2c --niter 1000000 --lr 0.0006 --nproc 8 --train_freq 16 --batch_size 128 --train_start 0 --save_dir breakout_a2c --checkpoint_freq 250000
