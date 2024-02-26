#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=extract_tseries
#SBATCH --output=/project/rrg-pbellec/mstlaure/cneuromod_extract_tseries/slurm_files/slurm-%A_%a.out
#SBATCH --error=/project/rrg-pbellec/mstlaure/cneuromod_extract_tseries/slurm_files/slurm-%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marie.stl@gmail.com

# load modules required for your script to work
module load StdEnv/2023
module load python/3.10.13

# activate project's virtual env
source /home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/env/bin/activate

SUBJECT="${1}"  # e.g., "01"
CODEDIR="/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries"
DATADIR="${CODEDIR}/data/hcptrt.fmriprep"
OUTDIR="${CODEDIR}/masks/yeo-7net"
ATLAS="tpl-MNI152NLin2009bSym/tpl-MNI152NLin2009bSym_res-03_atlas-MIST_desc-ROI_dseg.nii.gz"

# launch job
python seed_connectivity.py \
    --data_dir "${DATADIR}" \
    --out_dir "${OUTDIR}" \
    --atlas_dir "${CODEDIR}/atlases" \
    --atlas "${ATLAS}" \
    --subject "${SUBJECT}"
