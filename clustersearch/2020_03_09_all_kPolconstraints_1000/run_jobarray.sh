#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -N 1                               # Request one node (if you request more than one core with -c, also using
                                           # -N 1 means all cores will be on the same node)
#SBATCH --array 1-528:1
#SBATCH -t 8-00:00                         # Runtime in D-HH:MM format
#SBATCH -p long                           # Partition to run in
#SBATCH --mem=700                    # Memory total in MB (for all cores)
#SBATCH -o %A_%a.out                 # File to which STDOUT will be written, including job ID
#SBATCH -e %A_%a.err                 # File to which STDERR will be written, including job ID
#SBATCH --mail-type=NONE                    # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=abc123@hms.harvard.edu   # Email to which notifications will be sent
#SBATCH -J c2syn

module load gcc/6.2.0
module load python/3.6.0
#export LD_LIBRARY_PATH=/n/app/gsl/2.3/lib do not use this or it will not find python
source /home/rm335/myenvpy36/bin/activate

python boundariessyn_variousconditions_kpc.py $SLURM_ARRAY_TASK_ID
