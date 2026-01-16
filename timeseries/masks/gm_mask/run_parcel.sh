#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=extract_tseries
#SBATCH --output=/project/rrg-pbellec/mstlaure/cneuromod_extract_tseries/slurm_files/slurm-%A_%a.out
#SBATCH --error=/project/rrg-pbellec/mstlaure/cneuromod_extract_tseries/slurm_files/slurm-%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marie.stl@gmail.com

# load modules required for your script to work
module load StdEnv/2023
module load python/3.10.13

# activate project's virtual env
source /home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/env/bin/activate

SUBJECT="${1}"  # e.g., "01"
SEASON="${2}"  # e.g., "06"
NPARCELS="${3}" # e.g., 1000

# launch job
python step5_downsample_parcellation.py \
    --subject "${SUBJECT}" \
    --space T1w \
    --nparcels "${NPARCELS}" \
    --season "${SEASON}"
