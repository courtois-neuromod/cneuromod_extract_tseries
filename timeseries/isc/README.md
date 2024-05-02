Inter-subject correlations scripts
===================

Collection of scripts to compute voxelwise inter-subject correlations on the CNeuroMod Friends dataset. Adapted from [cneuromod_alpha2_ISC](https://github.com/courtois-neuromod/cneuromod_alpha2_ISC/tree/master).

Source: ``./timeseries/isc`` \
Scripts:

* ``step1_preprocess_data.py``. Script denoises and masks ``*bold.nii.gz`` files, saves them in one .hdf5 file per subject.
* ``step2_extract_isc.py``. Script computes correlations.
* ``step3_visualize.py``. TBD.
