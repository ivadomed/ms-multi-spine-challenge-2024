#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --job-name=challengejob35x36x37x38     # set a more descriptive job-name
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=1-00:00:00   # DD-HH:MM:SS
#SBATCH --output=/home/p/plb/links/scratch/challenge/%x_%A_v2.out
#SBATCH --error=/home/p/plb/links/scratch/challenge/%x_%A_v2.err
#SBATCH --mail-user=pierrelouis.benveniste03@gmail.com     # whenever the job starts/fails/completes, an email will be sent 
#SBATCH --mail-type=ALL

# Launch jobs
parallel --verbose --jobs 4 ::: \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/challenge/ms-multi-spine-challenge-2024/finetuning_experiment/compute_canada_script/continue_job35.sh 2>&1 | tee /home/p/plb/links/scratch/challenge/logfile_job35_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/challenge/ms-multi-spine-challenge-2024/finetuning_experiment/compute_canada_script/continue_job36.sh 2>&1 | tee /home/p/plb/links/scratch/challenge/logfile_job36_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/challenge/ms-multi-spine-challenge-2024/finetuning_experiment/compute_canada_script/continue_job37.sh 2>&1 | tee /home/p/plb/links/scratch/challenge/logfile_job37_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/challenge/ms-multi-spine-challenge-2024/finetuning_experiment/compute_canada_script/continue_job38.sh 2>&1 | tee /home/p/plb/links/scratch/challenge/logfile_job38_\$ts.txt)" \