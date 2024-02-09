module load StdEnv/2020
module load gcc/9.3.0
module load ants/2.3.5

# Instructions to convert MNI parcels to T1w space with fmriprep output:
# https://neurostars.org/t/how-to-transform-mask-from-mni-to-native-space-using-fmriprep-outputs/2880/8

PARCELPATH="/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/masks/yeo_networks/parcel_masks"  # /path/to/infile_MNI.nii
SPREPDIR="/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/data/friends.fmriprep/sourcedata/smriprep"

for PNAME in default-mode dorsal_attention fronto-parietal sensorimotor ventral_attention visual
do
  for SUBNUM in 01 02 03 04 05 06
  do
    antsApplyTransforms --default-value 0 --dimensionality 3 --float 0 \
    --input ${PARCELPATH}/sub-${SUBNUM}_${PNAME}_parcel_MNI.nii.gz --interpolation NearestNeighbor \
    --output ${PARCELPATH}/sub-${SUBNUM}_${PNAME}_parcel_T1w.nii.gz \
    --reference-image ${SPREPDIR}/sub-${SUBNUM}/anat/sub-${SUBNUM}_desc-preproc_T1w.nii.gz \
    --transform ${SPREPDIR}/sub-${SUBNUM}/anat/sub-${SUBNUM}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5
  done
done

echo "Job finished"
