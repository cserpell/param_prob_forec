#!/bin/bash
#SBATCH -J series
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -o example_%j.out
#SBATCH -e example_%j.err
#SBATCH -t 100:00:00
#SBATCH --array=1-100
#SBATCH --mem=16000

use anaconda3
. p3/bin/activate

./run_expers_distribs_in_out.sh

