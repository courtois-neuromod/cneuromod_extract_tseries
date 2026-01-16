#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=extract_tseries
#SBATCH --output=/project/rrg-pbellec/mstlaure/cneuromod_extract_tseries/slurm_files/slurm-%A_%a.out
#SBATCH --error=/project/rrg-pbellec/mstlaure/cneuromod_extract_tseries/slurm_files/slurm-%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marie.stl@gmail.com

# load modules required for your script to work
module load StdEnv/2023
module load python/3.10.13

# activate project's virtual env
source /home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/env/bin/activate

IDIR="/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/isc"

SEASON="${1}"  # e.g., "06"
SUBJECTS="01 02 03 04 05 06" # e.g., "01 02 03"

# launch job
python step2_extract_isc.py \
    --subjects ${SUBJECTS} \
    --idir "${IDIR}" \
    --space MNI152NLin2009cAsym \
    --use_simple \
    --season "${SEASON}"

