#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like so ##SBATCH
#SBATCH --partition main                         ### specify partition name where to run a job. main - 7 days time limit
#SBATCH --time 7-00:00:00                      ### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name grid_search                   ### name of the job. replace my_job with your desired job name
#SBATCH --output grid_search-id-%J.out                ### output log for running job - %J is the job number variable                         ### job array index
##SBATCH --mail-user=yaararum@post.bgu.ac.il      ### users email for sending job status notifications
##SBATCH --mail-type=FAIL             ### conditions when to send the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --cpus-per-task=4	# 20 cpus per task – use for multithreading, usually with --tasks=1
#SBATCH --mem-per-cpu=1G
##SBATCH --mem=1G
#SBATCH --tasks=1		# 1 processes – use for multiprocessing 

### Print some data to output file ###
echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST

### Start you code below 	####
module load anaconda              	### load anaconda module
source activate yaara_env            	### activating Conda environment, environment must be configured before running the job

python NoExpSTSRHOSPD.py --model=1