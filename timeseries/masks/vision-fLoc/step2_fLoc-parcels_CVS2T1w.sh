module load StdEnv/2020
module load gcc/9.3.0
module load fsl/6.0.3

module load freesurfer/7.1.1
module load ants/2.3.5

source /project/rrg-pbellec/mstlaure/.virtualenvs/things_memory_results/bin/activate

# Loading the fsl module sets up $FSLDIR, no need to define it

# by default, my $SUBJECTS_DIR is in my home directory on beluga :
# /home/mstlaure/.local/easybuild/software/2020/Core/freesurfer/7.1.1/subjects

# CVS parcels dowloaded from Kanwisher group at:
# https://web.mit.edu/bcs/nklab/GSS.shtml#download

# Instructions to convert CSV parcels to mni from:
# https://neurostars.org/t/freesurfer-cvs-avg35-to-mni-registration/17581

# Instructions to convert MNI parcels to T1w space with fmriprep output:
# https://neurostars.org/t/how-to-transform-mask-from-mni-to-native-space-using-fmriprep-outputs/2880/8

# Only need to do this once, already done...
# Generates reg.mni152.2mm.dat file under $SUBJECTS_DIR/cvs_avg35/mri/transforms/
#mni152reg --s cvs_avg35

PARCELDIR="/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/masks/vision-fLoc"
SPREPDIR="/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/data/friends.fmriprep/sourcedata/smriprep"

for PARAM in bodyEBA faceFFA faceOFA facepSTS sceneMPA sceneOPA scenePPA
do
  mri_vol2vol --targ "${PARCELDIR}/tpl-CVSavg35/tpl-CVSavg35_atlas-fLocVisionKanwisher_label-${PARAM}_mask.nii" \
  --mov $FSLDIR/data/standard/MNI152_T1_2mm.nii.gz \
  --o ${PARCELDIR}/tpl-MNIT1/tpl-MNIT1_res-2mm_atlas-fLocVisionKanwisher_label-${PARAM}_pseg.nii \
  --inv --reg ${SUBJECTS_DIR}/cvs_avg35/mri/transforms/reg.mni152.2mm.dat

  mri_vol2vol --targ "${PARCELDIR}/tpl-CVSavg35/tpl-CVSavg35_atlas-fLocVisionKanwisher_label-${PARAM}_desc-L_mask.nii" \
  --mov $FSLDIR/data/standard/MNI152_T1_2mm.nii.gz \
  --o "${PARCELDIR}/tpl-MNIT1/tpl-MNIT1_res-2mm_atlas-fLocVisionKanwisher_label-${PARAM}_desc-L_pseg.nii" \
  --inv --reg ${SUBJECTS_DIR}/cvs_avg35/mri/transforms/reg.mni152.2mm.dat

  mri_vol2vol --targ "${PARCELDIR}/tpl-CVSavg35/tpl-CVSavg35_atlas-fLocVisionKanwisher_label-${PARAM}_desc-R_mask.nii" \
  --mov $FSLDIR/data/standard/MNI152_T1_2mm.nii.gz \
  --o "${PARCELDIR}/tpl-MNIT1/tpl-MNIT1_res-2mm_atlas-fLocVisionKanwisher_label-${PARAM}_desc-R_pseg.nii" \
  --inv --reg ${SUBJECTS_DIR}/cvs_avg35/mri/transforms/reg.mni152.2mm.dat

  for SUBNUM in 01 02 03 04 05 06
  do
    antsApplyTransforms --default-value 0 --dimensionality 3 --float 0 \
    --input ${PARCELDIR}/tpl-MNIT1/tpl-MNIT1_res-2mm_atlas-fLocVisionKanwisher_label-${PARAM}_pseg.nii --interpolation Linear \
    --output ${PARCELDIR}/tpl-sub${SUBNUM}T1w/tpl-sub${SUBNUM}T1w_res-anat_atlas-fLocVisionKanwisher_label-${PARAM}_pseg.nii \
    --reference-image ${SPREPDIR}/sub-${SUBNUM}/anat/sub-${SUBNUM}_desc-preproc_T1w.nii.gz \
    --transform ${SPREPDIR}/sub-${SUBNUM}/anat/sub-${SUBNUM}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5

    antsApplyTransforms --default-value 0 --dimensionality 3 --float 0 \
    --input "${PARCELDIR}/tpl-MNIT1/tpl-MNIT1_res-2mm_atlas-fLocVisionKanwisher_label-${PARAM}_desc-L_pseg.nii" --interpolation Linear \
    --output "${PARCELDIR}/tpl-sub${SUBNUM}T1w/tpl-sub${SUBNUM}T1w_res-anat_atlas-fLocVisionKanwisher_label-${PARAM}_desc-L_pseg.nii" \
    --reference-image ${SPREPDIR}/sub-${SUBNUM}/anat/sub-${SUBNUM}_desc-preproc_T1w.nii.gz \
    --transform ${SPREPDIR}/sub-${SUBNUM}/anat/sub-${SUBNUM}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5

    antsApplyTransforms --default-value 0 --dimensionality 3 --float 0 \
    --input "${PARCELDIR}/tpl-MNIT1/tpl-MNIT1_res-2mm_atlas-fLocVisionKanwisher_label-${PARAM}_desc-R_pseg.nii" --interpolation Linear \
    --output "${PARCELDIR}/tpl-sub${SUBNUM}T1w/tpl-sub${SUBNUM}T1w_res-anat_atlas-fLocVisionKanwisher_label-${PARAM}_desc-R_pseg.nii" \
    --reference-image ${SPREPDIR}/sub-${SUBNUM}/anat/sub-${SUBNUM}_desc-preproc_T1w.nii.gz \
    --transform ${SPREPDIR}/sub-${SUBNUM}/anat/sub-${SUBNUM}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5
  done
done

echo "Job finished"
