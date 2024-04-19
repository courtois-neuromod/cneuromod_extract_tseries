
module load StdEnv/2020
module load gcc/9.3.0
module load ants/2.3.5

DATADIR="/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/masks/gm-masks/freesurfer"
SPREPDIR="/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/data/friends.fmriprep/sourcedata/smriprep"

for SUBNUM in 01 02 03 04 05 06
do
  antsApplyTransforms --default-value 0 --dimensionality 3 --float 0 \
  --input ${DATADIR}/sub-${SUBNUM}_space-T1w_label-GM_desc-float_dseg.nii.gz --interpolation Linear \
  --output ${DATADIR}/sub-${SUBNUM}_space-MNI152NLin2009cAsym_label-GM_desc-FSLinear_probseg.nii.gz \
  --reference-image ${SPREPDIR}/sub-${SUBNUM}/anat/sub-${SUBNUM}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz \
  --transform ${SPREPDIR}/sub-${SUBNUM}/anat/sub-${SUBNUM}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5
done
