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
For each subject, individual ROI masks and parcellations are saved under ``atlases/tpl-sub0*T1w/``

Those include:

**Language ROI masks (n=8) from Mariya Toneva and Leila Wehbe**

* Angular gyri (AngularG)
* Anterior and posterior temporal cortex (AntTemp, PostTemp)
* Dorsomedial prefrontal cortex (dmpfc)
* Inferior frontal gyri (IFG and IFGorb)
* Middle frontal gyri (MFG)
* Posterior cingulate cortex (pCingulate)

For each subject, language ROIS were warped from standard MNI space to
individual space. These ROIs are based on the work of Fedorenko et al. 2010
and Binder et al., 2009, as explained `here <https://www.biorxiv.org/content/10.1101/2020.09.28.316935v4>`_


**Yeo networks (n=6) derived from seed functional connectivity**

* Default Mode network
* Dorsal Attention network
* Front-parietal network
* Sensorimotor network
* Ventral attention network
* Visual network

Networks are based on `Yeo et al., 2011 <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3174820/>`_
For each subject, each network was derived by averaging the signal within a
seed parcel (MIST-ROI atlas), and by correlating its activation with the
rest of the brain using resting state runs from the hcptrt dataset.

**Early visual ROIs derived from retinotopy**

* V1, V2, V3, V3a, V3b, VO1, VO2, hV4, LO1, LO2, TO1 and TO2

For three subjects (sub-01, sub-02 and sub-03) who completed a retinotopy task,
ROI masks from the early visual cortex were derived from their population
receptive fields and from group priors using the `Neuropythy toolbox <https://github.com/noahbenson/neuropythy>`_.

**Higher visual ROIs derived from fLoc**

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
own data. For all subjects, group parcels of regions with face, scene and
body preferences identified by the `Kanwisher lab <https://web.mit.edu/bcs/nklab/GSS.shtml#download>`_ were also warped into
single-subject space.
