"""

Code to produce ROI masks in single-subject space,
or single-subject masks in MNI space from functional data (not from template)


Vision: localisers in 3 out of 6 subjects for floc and retino in T1w space
Use group atlases for other subjects?

Audio: Maelle used MIST (MNI space)

Language: ask Valentina and Maria
some from HCPtrt task... Federenko localizers


Functional networks
FP code: in MNI space
derive single-subject networks that can be converted into binary masks
based on seed coordinates from Yeo's 7 functional networks

Links:
- https://github.com/courtois-neuromod/fmri-generator/blob/master/scripts/seed_connectivity.py
- https://nilearn.github.io/stable/auto_examples/03_connectivity/plot_seed_to_voxel_correlation.html
"The maps presented in the results are using seeds in 6 of the 7 Yeo networks [30]: the default-mode network with a seed in the cingulate gyrus (-7, -52, 26), the visual network with a seed in the superior lingual gyrus (-16, -74, 7), the sensorimotor network with a seed in the postcentral gyrus (-41, -20, 62), the dorsal attentional network with another seed in the postcentral gyrus (-34, -38, 44), the ventral attentional network with a seed in the cingulate gyrus (-5, 15, 32) and the fronts-parietal network with a seed in the middle frontal gyrus (-40, 50, 7). The seventh limbic network was excluded because it is composed of regions with high signal loss and distortions due to field inhomogeneity."





converting MNI coordinates (in mm) to T1W coordinates (in voxels)

USE FSL FLIRT std2imgcoord, combine two transformations...
https://neurostars.org/t/coordinates-from-mni-to-subject-space-using-transform-h5-file/5994/3

https://neurostars.org/t/mni-coordinates-to-subjects-epi/5628

https://neurostars.org/t/mni-to-native-space/2107/2

https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT/UserGuide#img2imgcoord
"""
import glob
import nilearn
import nibabel as nib

found_mask_list = sorted(
    glob.glob(
        f"{self.bids_dir}/sub-{subject}/"
        f"ses-*/func/*{self.config.template}"
        "*_mask.nii.gz",
        ),
    )
if exclude := utils._check_mask_affine(found_mask_list):
    found_mask_list, __annotations__ = utils._get_consistent_masks(
        found_mask_list,
        exclude,
        )
    print(f"Remaining: {len(found_mask_list)} masks")

bold_list = []
mask_list = []
for fm in found_mask_list:
    identifier = fm.split('/')[-1].split('_space')[0]
    sub, ses = identifier.split('_')[:2]

    bpath = sorted(glob.glob(
        f"{self.bids_dir}/{sub}/{ses}/func/{identifier}"
        f"*{self.config.template}*_desc-preproc_*bold.nii.gz"
    ))

    if len(bpath) == 1 and Path(bpath[0]).exists():
        bold_list.append(bpath[0])
        mask_list.append(fm)

    subject_epi_mask = compute_multi_epi_mask(
        mask_list,
        lower_cutoff=0.2,
        upper_cutoff=0.85,
        connected=True,
        opening=False,  # we should be using fMRIPrep masks
        threshold=0.5,
        target_affine=None,
        target_shape=None,
        exclude_zeros=False,
        n_jobs=1,
        memory=None,
        verbose=0,
    )



        parcel = nib.load(par)
        gm_mask = nib.load(gm)
        rs_parcel = nilearn.image.resample_to_img(parcel, gm_mask, interpolation='continuous')

        rs_parcel = nib.nifti1.Nifti1Image((rs_parcel.get_fdata() > 0.5).astype(int), affine=rs_parcel.affine)
