#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=deepgaze
#SBATCH --output=/project/rrg-pbellec/mstlaure/friends_algonauts/slurm_files/slurm-%A_%a.out
#SBATCH --error=/project/rrg-pbellec/mstlaure/friends_algonauts/slurm_files/slurm-%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marie.stl@gmail.com

# load modules
module load StdEnv/2020
module load python/3.8.2

# activate project's virtual env
source /home/mstlaure/projects/rrg-pbellec/mstlaure/friends_algonauts/fa_venv/bin/activate

INPUT="/home/mstlaure/projects/rrg-pbellec/mstlaure/friends_algonauts/processed_data/RR_features"
BIDSDIR="/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/outputs/ood"
OUTPUT="/project/rrg-pbellec/mstlaure/friends_algonauts/results/RR_scores_pipelineTest"

SUBJECT_NUM="${1}" # 01, 02, 03
ATLAS="${2}" # MIST, Schaefer18
PARCEL="${3}" # 444, 1000Parcels7Networks

# launch job
python -m model_ridgereg_plus_ood \
        --idir ${INPUT} \
        --bdir ${BIDSDIR} \
        --odir ${OUTPUT} \
        --participant "sub-${SUBJECT_NUM}" \
        --atlas "${ATLAS}" \
        --parcel "${PARCEL}" \
        --modalities text \
        --text_features text_pooled text_token text_1hot \
        --back 5 \
        --input_duration 3 \
        --verbose 1
