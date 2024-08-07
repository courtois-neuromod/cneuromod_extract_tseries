Available Parcellations
=======================

1. Standard parcellations
-------------------------

**MIST and BASC atlases**

* MIST and BASC parcellations are under ``atlases/tpl-MNI152NLin2009bSym``
* BASC parcellations are also under ``atlases/tpl-MNI152NLin2009bAsym``

**Other standard atlases in MNI152NLin2009cAsym space**

* The Schaefer and DiFuMo parcellations are under ``atlases/tpl-MNI152NLin2009cAsym/``


2. Subject-specific parcellations
---------------------------------
For each subject, individual ROI masks and parcellations in native space are saved under ``atlases/tpl-sub0*T1w/``

Those include:

**Individual grey matter masks**

* Subject-specific masks that include cortical and subcortical grey matter were created from Freesurfer segmentation (aseg.mgz) in T1w space. E.g. ``atlases/tpl-sub0*T1w/tpl-sub*T1w_res-func_label-GM_desc-fromFS_dseg.nii.gz``
* Individual grey matter masks warped to MNI space are also available as ``atlases/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_sub-*_res-func_label-GM_desc-fromFS_dseg.nii.gz``
* Individual 10k-parcel parcellations performed with the Ward algorithm within individual grey matter masks in subject (T1w) or MNI space also provide features that cover the entire brain at a resolution lower than EPI voxels. Each of the 10k parcels is nested within the Schaefer18_1000Parcels7Networks parcel with which is shares the most voxels, for a nested hierarchical parcellation. Note that, in T1w space, the Schaefer parcellation was warped to subject space to support this hierarchical assignment. Also, some unassigned parcels fall within the cerebellum or subcortical grey matter structures, regions not covered by the Schaefer parcellation. E.g., ``atlases/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_sub-*_res-func_atlas-Ward_desc-10k_dseg.nii.gz``, ``atlases/tpl-sub*T1w/tpl-sub*T1w_res-func_atlas-Ward_desc-10k_dseg.nii.gz``. The nested assignment to larger Shaefer18-1000Parcels7Networks is detailed in ``atlases/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_sub-*_atlas-Ward_desc-10k_dseg.tsv`` and ``atlases/tpl-sub*T1w/tpl-sub*T1w_atlas-Ward_desc-10k_dseg.tsv``.


**Language ROI masks (n=8) from Mariya Toneva and Leila Wehbe**

ROIs include:

* Angular gyri (AngularG)
* Anterior and posterior temporal cortex (AntTemp, PostTemp)
* Dorsomedial prefrontal cortex (dmpfc)
* Inferior frontal gyri (IFG and IFGorb)
* Middle frontal gyri (MFG)
* Posterior cingulate cortex (pCingulate)

E.g., ``atlases/tpl-sub0*T1w/tpl-sub*T1w_res-func_atlas-langToneva_label-AntTemp_mask.nii.gz``

For each subject, language ROIS were warped from standard MNI space to
individual space. These ROIs are based on the work of Fedorenko et al. 2010
and Binder et al., 2009, as explained `here <https://www.biorxiv.org/content/10.1101/2020.09.28.316935v4>`_

ROI masks are also available in MNI space in a format compatible with all 6 subjects.
E.g., ``atlases/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-func_atlas-langToneva_label-AntTemp_mask.nii.gz``


**Yeo networks (n=6) derived from seed functional connectivity**

Networks include:

* Default Mode network
* Dorsal Attention network
* Front-parietal network
* Sensorimotor network
* Ventral attention network
* Visual network

Networks are based on `Yeo et al., 2011 <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3174820/>`_
For each subject, each network was derived by averaging the signal within a
seed parcel (MIST-ROI atlas), and by correlating its activation with the
rest of the brain using resting state runs from the hcptrt dataset. Network masks are available in
MNI and native (T1w) space for all subjects.

E.g., ``atlases/tpl-sub0*T1w/tpl-sub*T1w_res-func_atlas-yeo7networks_label-defaultMode_mask.nii.gz``,
``atlases/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_sub-0*_res-func_atlas-yeo7networks_label-defaultMode_mask.nii.gz``


**Early visual ROIs derived from retinotopy**

ROIs include:

* V1, V2, V3, V3a, V3b, VO1, VO2, hV4, LO1, LO2, TO1 and TO2

For four subjects (sub-01, sub-02, sub-03 and sub-05) who completed a retinotopy task,
ROI masks from the early visual cortex were derived from their population
receptive fields and from group priors using the `Neuropythy toolbox <https://github.com/noahbenson/neuropythy>`_.
ROI masks are available in MNI and native (T1w) space for the four subjects who completed the retinotopy task.

E.g., ``atlases/tpl-sub0*T1w/tpl-sub*T1w_res-{anat, func}_atlas-retinoVisionNpythy_label-V1_mask.nii.gz``,
``atlases/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_sub-0*_res-{func, anat}_atlas-retinoVisionNpythy_label-V1_mask.nii.gz``


**Higher visual ROIs derived from fLoc**

ROIs include:

* Extrastriate body area (body-EBA)
* Fusiform face area (face-FFA)
* Occipital face area (face-OFA)
* Posterior superior temporal sulcus (face-pSTS)
* Medial place area (scene-MPA)
* Occipital place area (scene-OPA)
* Parahippocampal place area (scene-PPA)

For three subjects (sub-01, sub-02 and sub-03) who completed the fLoc task,
ROI masks from higher level visual areas with face, scene and
body preferences were identified with a combination of group priors and their
own data. Binary ROI masks are available in MNI and native (T1w) space.

E.g., ``atlases/tpl-sub0*T1w/tpl-sub*T1w_res-func_atlas-fLocVisionTask_label-faceFFA_mask.nii.gz``,
``atlases/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_sub-0*_res-func_atlas-fLocVisionTask_label-faceFFA_mask.nii.gz``

For all subjects, group parcels of regions with face, scene and
body preferences identified by the `Kanwisher lab <https://web.mit.edu/bcs/nklab/GSS.shtml#download>`_ were also warped into
single-subject space.

E.g., ``atlases/tpl-sub0*T1w/tpl-sub*T1w_res-func_atlas-fLocVisionKanwisher_label-faceFFA_mask.nii.gz``

The Kanwisher lab parcels are also available in MNI space in a standardized format that fits all subjects.

E.g., ``atlases/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-func_atlas-fLocVisionKanwisher_label-faceFFA_mask.nii.gz``
