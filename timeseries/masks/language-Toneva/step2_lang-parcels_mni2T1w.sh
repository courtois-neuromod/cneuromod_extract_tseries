module load python/3.7.9
module load StdEnv/2020
module load gcc/9.3.0
module load ants/2.3.5

# Instructions to convert MNI parcels to T1w space with fmriprep output:
# https://neurostars.org/t/how-to-transform-mask-from-mni-to-native-space-using-fmriprep-outputs/2880/8

PARCELPATH="/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/masks/language-Toneva"  # /path/to/infile_MNI.nii
#SPREPDIR="/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/data/friends.fmriprep/sourcedata/smriprep"
SPREPDIR="/home/mstlaure/projects/rrg-pbellec/mstlaure/things_memory_results/data/things.fmriprep/sourcedata/smriprep"

for SUBNUM in 01 02 03 04 05 06
do
  antsApplyTransforms --default-value 0 --dimensionality 3 --float 0 \
  --input ${PARCELPATH}/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-func_atlas-langToneva_desc-8_dseg.nii.gz \
  --interpolation NearestNeighbor \
  --output ${PARCELPATH}/tpl-sub${SUBNUM}T1w/tpl-sub${SUBNUM}T1w_res-anat_atlas-langToneva_desc-8_dseg.nii.gz \
  --reference-image ${SPREPDIR}/sub-${SUBNUM}/anat/sub-${SUBNUM}_desc-preproc_T1w.nii.gz \
  --transform ${SPREPDIR}/sub-${SUBNUM}/anat/sub-${SUBNUM}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5

  for PNAME in pCingulate dmpfc PostTemp AntTemp AngularG IFG MFG IFGorb
  do
    antsApplyTransforms --default-value 0 --dimensionality 3 --float 0 \
    --input ${PARCELPATH}/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-func_atlas-langToneva_label-${PNAME}_mask.nii.gz \
    --interpolation NearestNeighbor \
    --output ${PARCELPATH}/tpl-sub${SUBNUM}T1w/tpl-sub${SUBNUM}T1w_res-anat_atlas-langToneva_label-${PNAME}_mask.nii.gz \
    --reference-image ${SPREPDIR}/sub-${SUBNUM}/anat/sub-${SUBNUM}_desc-preproc_T1w.nii.gz \
    --transform ${SPREPDIR}/sub-${SUBNUM}/anat/sub-${SUBNUM}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5
  done
done

echo "Job finished"
