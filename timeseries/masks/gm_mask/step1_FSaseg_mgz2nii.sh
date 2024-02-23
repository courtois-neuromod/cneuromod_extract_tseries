
module load freesurfer/7.1.1

# Instructions to convert .mgz freesurfer files to .nii.gz
# https://neurostars.org/t/freesurfer-mgz-to-nifti/21647

INDIR="/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/data/friends.fmriprep/sourcedata/smriprep/sourcedata/freesurfer"
OUTDIR="/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/masks/gm-masks/freesurfer"

for SUBNUM in 01 02 03 04 05 06
do
  mri_convert $INDIR/sub-${SUBNUM}/mri/aseg.mgz $OUTDIR/sub-${SUBNUM}_aseg.nii.gz
done

echo "Job finished"
