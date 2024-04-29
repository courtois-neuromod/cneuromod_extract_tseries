
module load StdEnv/2020
module load gcc/9.3.0
module load ants/2.3.5

ATLASDIR="/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/atlases/tpl-MNI152NLin2009cAsym"
DATADIR="/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/masks/gm-masks/parcellation"
SPREPDIR="/home/mstlaure/projects/rrg-pbellec/mstlaure/things_memory_results/data/things.fmriprep/sourcedata/smriprep"

for SUBNUM in 01 02 03 04 05 06
do
  antsApplyTransforms --default-value 0 --dimensionality 3 --float 0 \
  --input ${ATLASDIR}/tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-1000Parcels7Networks_dseg.nii.gz \
  --interpolation NearestNeighbor \
  --output ${DATADIR}/tpl-sub${SUBNUM}T1w_res-anat_atlas-Schaefer2018_desc-1000Parcels7Networks_dseg.nii.gz \
  --reference-image ${SPREPDIR}/sub-${SUBNUM}/anat/sub-${SUBNUM}_desc-preproc_T1w.nii.gz \
  --transform ${SPREPDIR}/sub-${SUBNUM}/anat/sub-${SUBNUM}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5
done
