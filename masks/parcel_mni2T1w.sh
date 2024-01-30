module load StdEnv/2020
module load gcc/9.3.0
module load fsl/6.0.3

module load freesurfer/7.1.1
module load ants/2.3.5

source /project/rrg-pbellec/mstlaure/.virtualenvs/things_memory_results/bin/activate

# Loading the fsl module sets up $FSLDIR, no need to define it

# by default, my $SUBJECTS_DIR is in my home directory on beluga :
# /home/mstlaure/.local/easybuild/software/2020/Core/freesurfer/7.1.1/subjects

# Instructions to convert MNI parcels to T1w space with fmriprep output:
# https://neurostars.org/t/how-to-transform-mask-from-mni-to-native-space-using-fmriprep-outputs/2880/8

MNIPARCEL="${1}"  # /path/to/infile_MNI.nii
OUTFILE="${2}"  # /path/to/outfile_T1w.nii
SPREPDIR="/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/data/friends.fmriprep/sourcedata/smriprep"

#PARCELDIR="/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/masks/standard_masks/fLoc_kanwisher_parcels"
#SUBPARCELDIR="/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/masks/subject_masks/fLoc"

for SUBNUM in 01 02 03 04 05 06
do
  antsApplyTransforms --default-value 0 --dimensionality 3 --float 0 \
  --input ${MNIPARCEL} --interpolation Linear \
  --output ${OUTFILE} \
  --reference-image ${SPREPDIR}/sub-${SUBNUM}/anat/sub-${SUBNUM}_desc-preproc_T1w.nii.gz \
  --transform ${SPREPDIR}/sub-${SUBNUM}/anat/sub-${SUBNUM}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5
done

echo "Job finished"
