Available Parcellations
=======================

TODO: describe available template parcellations (MNI space)
TODO: describe available subject-specific parcellations (MNI and T1w)






5. Standarization and smoothing
-------------------------------

Options for data standardization include ``zscore`` and ``psc``.
By default, the ``zscore`` standardization is called from the base config file
``cneuromod_extract_tseries/timeseries/config/base.yaml``. You can override this choice
at the command line.
E.g.,
:
    python run.py dataset=friends parcellation=mist444 standardize=psc

The default smoothing kernel size is set to 5.0 in the base config file.
You can specify a different value.
E.g.,
:
    python run.py dataset=friends parcellation=mist444 smoothing_fwhm=3.0


6. Parcellation
---------------

Config files that specify the timeseries parcellation parameters are saved under
``cneuromod_extract_tseries/timeseries/config/parcellation/<parcelation_name>.yaml``

The parcellation must be specified when launching the script.
:
    python run.py dataset=mario3 parcellation=parcelation_name

Custom parcellations can be added by creating <parcelation_name>.yaml files in
``cneuromod_extract_tseries/timeseries/config/parcellation`` and saving the atlas
needed to mask the signal (see below).


To use a *standard template atlas* (e.g., MIST, Schaefer 7 networks, DiFuMo),
the following parameters need to be specified in the parcellation .yaml file:

* ``template``. This field specifies whether to process fMRI data in native (T1w)
or in normalized (MNI) space. Specify ``template = MNI152NLin2009cAsym`` when
using a standard template.
* ``template_gm_path``. The path to a normalized grey matter mask. Grey matter
masks from the MNI152NLin2009cAsym template, which match the normalized
CNeuroMod data, are saved under
``cneuromod_extract_tseries/atlases/tpl-MNI152NLin2009cAsym``.
Recommended = ``tpl-MNI152NLin2009cAsym_res-02_label-GM_probseg.nii.gz``
* ``n_iter``. The number of iterations to perform a binary closing to merge the
template grey matter mask (specified with ``template_gm_path``) with a grey matter
mask derived from the subject's functional runs (recommended ``n_iter = 2``).
* ``parcel_name``. The name given to the parcellation. For each subject, a
subject-specific parcellation will be generated and saved as
<output_dir>/<dset_name>/subject_masks/<subject>_<template>_<parcel_name>.nii.gz.
* ``parcel_type``. Whether the specified template parcellation is discrete or
probabilistic. Choices = [``dseg``, ``probseg``]
* ``template_parcellation``. The full path to a standard atlas / parcellation
that specifies the ROI(s) from which to extract the timeseries.
Template parcellations should be saved under
 ``cneuromod_extract_tseries/atlases/tpl-<space>/tpl-<space>_<parcel_descript>.nii.gz``
 E.g.,
``cneuromod_extract_tseries/atlases/tpl-MNI152NLin2009bAsym/tpl-MNI152NLin2009bAsym_res-03_atlas-BASC_desc-444_dseg.nii.gz``

#TODO: look into templateflow...? save and pull options from repo?


To use a **custom subject-specific parcellation** (e.g., ROIs from a
functional localizer, network mask derived from seed-based functional
connectivity), whether in T1w or MNI space, the following parameters
need to be specified in the parcellation .yaml file:

* ``template = [T1w, MNI152NLin2009cAsym]``. This field specifies whether
to analyse fMRI data in native (T1w) or in normalized (MNI) space. Set it to
match the space of the subject-specific parcellation.
* ``template_gm_path``. The path to a normalized grey matter mask. Only needed
for analyses in MNI space (``template = MNI152NLin2009cAsym``), otherwise omit
from the parcellation config file or set to ``null``. Grey matter masks from the
MNI152NLin2009cAsym template, which match the normalized CNeuroMod data, are
provided under ``cneuromod_extract_tseries/atlases/tpl-MNI152NLin2009cAsym``.
Recommended = ``tpl-MNI152NLin2009cAsym_res-02_label-GM_probseg.nii.gz``.
* ``n_iter``. The number of iterations to perform a binary closing to merge the
template grey matter mask (specified with ``template_gm_path``) with a grey matter
mask derived from the subject's functional runs (recommended ``n_iter = 2``).
Only needed for analyses in MNI space (``template = MNI152NLin2009cAsym``),
otherwise omit this field from the config file or set it to ``null``.
* ``template_parcellation``. Set to ``null``.
* ``parcel_type``. Whether the specified template parcellation is discrete or
probabilistic. Choices = [``dseg``, ``probseg``].
* ``parcel_name``. The name of the parcellation. Custom subject-specific
parcellations (in T1w or MNI space) directly under
``<output_dir>/<dset_name>/subject_masks/<subject>_<template>_<parcel_name>.nii.gz``.
E.g., ``cneuromod_extract_tseries/output/friends/subject_masks/sub-01_T1w_<parcel_name>.nii.gz``

# TODO: list provided parcellations

TODO (optional):
- add your own custom parcellation: see masks.py (WIP), parcellations.rst (WIP)
