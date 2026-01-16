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
module load python/3.8.2

# activate project's virtual env
source /home/mstlaure/projects/rrg-pbellec/mstlaure/friends_algonauts/fa_venv/bin/activate

INPUT="/home/mstlaure/projects/rrg-pbellec/mstlaure/friends_algonauts/processed_data/RR_features"
BIDSDIR="/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/outputs/friends"
OUTPUT="/project/rrg-pbellec/mstlaure/friends_algonauts/results/RR_scores_pipelineTest"

SUBJECT_NUM="${1}" # 01, 02, 03
ATLAS="${2}"
PARCEL="${3}" # MIST_444, Schaefer18_1000N7, Schaefer18_400N7

# launch job
python -m model_ridgereg_plus_quicktest \
        --idir ${INPUT} \
        --bdir ${BIDSDIR} \
        --odir ${OUTPUT} \
        --participant "sub-${SUBJECT_NUM}" \
        --atlas "${ATLAS}" \
	--parcel "${PARCEL}" \
        --modalities visual audio text \
        --text_features text_pooled text_token text_OAI text_1hot \
        --back 5 \
        --input_duration 3 \
        --n_split 10 \
        --random_state 7 \
        --verbose 1
