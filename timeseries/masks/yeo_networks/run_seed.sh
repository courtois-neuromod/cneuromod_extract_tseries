#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=extract_tseries
#SBATCH --output=/project/rrg-pbellec/mstlaure/cneuromod_extract_tseries/slurm_files/slurm-%A_%a.out
#SBATCH --error=/project/rrg-pbellec/mstlaure/cneuromod_extract_tseries/slurm_files/slurm-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marie.stl@gmail.com

# load modules required for your script to work
module load StdEnv/2023
module load python/3.10.13

# activate project's virtual env
source /home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/env/bin/activate

SUBJECT=="${1}"  # e.g., "01"
CODEDIR="/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries"
DATADIR="${CODEDIR}/data/friends.fmriprep"
OUTDIR="${CODEDIR}/masks/yeo_networks"

# launch job
python seed_connectivity.py \
    --data_dir "${DATADIR}" \
    --out_dir "${OUTDIR}" \
    --task_filter "b_" \
    --subject="sub-${SUBJECT}"
