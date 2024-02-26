Mask-Making scripts
===================

Collection of scripts to generate ROI masks and parcellations to extract timeseries from CNeuroMod datasets.

Script output, including from intermediary steps, is saved under ./masks according to their source (e.g., language-Toneva, vision-fLoc, yeo-7net, etc.)

Final masks and atlases used for timeseries extraction are saved under ./atlases according to their
space (e.g.,tpl-MNI152NLin2009cAsym, tpl-sub01T1w for masks warped to sub-01's subject space, etc.)

Current scripts are for the following masks and atlases:


**0. Grey matter masks from Freesurfer**

Source: ``./timeseries/gm_mask`` \
Scripts:

* ``step1_FSaseg_mgz2nii.sh``. Script converts aseg.mgz segmentations from Freesurfer
into nii.gz files.
* ``step2_FSaseg_2_T1wGMmask.py``. Script creates subject grey matter masks in
T1w space from Freesurfer parcellation using cortical and subcortical grey
matter indices.
* ``step3_GMmask_T1w2MNI.sh``. Script uses ANTS and fmriprep transformation
matrices to warp grey matter masks from T1w to MNI space for each subject.
* ``step4_anat2EPI_GM-masks.py``. Script downsamples subject grey matter masks
in T1w and MNI space from anatomical to functional (EPI) resolution.



**1. fLoc Visual Localizer**

Source: ``./timeseries/vision-fLoc`` \
Scripts:

* ``split_fLoc_CVSparcels_perROI.py``. Script takes group parcels from the [Kanwisher lab](https://web.mit.edu/bcs/nklab/GSS.shtml#download) in cvs_avg35 space and generates a separate binary mask for each ROI.
* ``fLoc-parcels_CVS2T1w.sh``. Script takes group parcels in cvs_avg35 space, warps them to MNI space, then warps them to individual (T1w) space for each subject (result is probabilistic mask).
* ``fLoc_parcel2mask.py``. Script resamples subject-space probabilistic parcels to the subject's EPI (functional) space, thresholds the parcels to obtain binary masks.


**2. 8 Language ROIs (from Toneva & Wehbe)**

Source: ``./timeseries/language-Toneva`` \
Scripts:

* ``step1_format_language_rois.py``. Script saves the source parcellation atlas as a .nii file with the .affine matrix of functional .nii.gz files in MNI space for the CNeuroMod subjects. A binary mask in (functional) MNI space is also created for each ROI (n=8).
* ``step2_lang-parcels_mni2T1w.sh``. Script warps binary parcel masks from MNI space to individual (anatomical T1w) space for each CNeuroMod subject with ants using the fmri.prep transformation matrix.
* ``step3_format_language_rois.py``. Script resamples binary masks in individual T1w space from anatomical to functional (EPI) resolution. Output is final binary masks in EPI space.


**3. Six of seven Yeo et al. 2011 networks from functional connectivity**

Source: ``./timeseries/yeo-7networks`` \
Scripts:

* ``seed_connectivity.py``. If seed parcel masks do not exist in MNI and individual space, the script takes seed voxel coordinates in MNI space from 6 of 7 Yeo et al. (2011) networks, positions seeds within a MIST-ROI parcel, and generates a binary parcel mask. These masks can be warped to individual space with ``yeo-parcels_mni2T1w.sh``. If parcel masks exist in MNI and subject space, the script extracts their timeseries (from resting state runs in the hcptrt dataset), correlates them with the rest of the brain, averages these correlations across multiple runs (per subject), and thresholds the resulting maps to obtain binary network masks.
* ``yeo-parcels_mni2T1w.sh``. Script warps binary parcel masks from MNI space to individual (anatomical T1w) space for each CNeuroMod subject with ants using the fmri.prep transformation matrix.


TODO: Audio: Maelle used MIST (MNI space)
TODO: Vale's methods, use Federenko parcels from prob of 200 subjects and intersect w subjects' story > math map from hcptrt.
TODO: the retino ROIs are from the retinotopy project, already binary masks in EPI space from Neuropythy output
TODO Vision: retino, use group atlas for 3 remaining subjects?
