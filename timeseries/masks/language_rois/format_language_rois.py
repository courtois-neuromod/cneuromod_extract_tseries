"""
Language-relevant ROI masks in MNI space from Mariya Toneva

These ROI were chosen based on previous work showing that they're involved in
language processing, and we've also found in several studies that they're well
predicted by features from NLP models.
More info here:
https://www.biorxiv.org/content/10.1101/2020.09.28.316935v2)

Masks are saved as .nii files with the .affine matrix that corresponds to
functional data in MNI space for all CNeuroMod subjects.

This script exports a parcellation in mni space (with all 8 labels),
and individual binary masks in MNI space for each parcel.

"""
from pathlib import Path
import nibabel as nib
from nilearn.image import resample_to_img
import numpy as np

MNI_FUNC_AFFINE = np.array(
    [[2.0, 0.0, 0.0,  -96.5],
    [ 0.0, 2.0, 0.0, -132.5],
    [ 0.0, 0.0, 2.0,  -78.5],
    [ 0.0, 0.0, 0.0,   1.0]]
)

lang_dir = Path(
    "../../../masks/language_rois"
).resolve()

out_path = Path(
    f"{lang_dir}/ROI_parcels_mni/"
    "tpl-MNI152NLin2009cAsym_language-ROIs_Toneva-8_dseg.nii.gz"
)

roi_dict = {
    1: "pCingulate",
    2: "dmpfc",
    3: "PostTemp",
    4: "AntTemp",
    5: "AngularG",
    6: "IFG",
    7: "MFG",
    8: "IFGorb",
}

if not out_path.exists():
    file_path = Path(
        f"{lang_dir}/from_Toneva/"
        "MNI_language_roi_downsampled_to_mni_t1_2mm_brain_4_8_21.nii"
    )

    Path(f"{lang_dir}/ROI_parcels_mni").mkdir(parents=True, exist_ok=True)
    lang_parcels = nib.load(file_path)

    lang_parcels_mni = nib.nifti1.Nifti1Image(
        lang_parcels.get_fdata().astype(int),
        affine=MNI_FUNC_AFFINE,
        dtype="uint8",
    )
    lang_parcels_mni.to_filename(f"{lang_dir}/ROI_parcels_mni/tpl-MNI152NLin2009cAsym_language-ROIs_Toneva-8_dseg.nii.gz")


    """
    Split parcellation into separate parcel masks in MNI space
    """

    Path(f"{lang_dir}/binary_masks_mni").mkdir(parents=True, exist_ok=True)

    for val, roi_name in roi_dict.items():
        mask_arr = (lang_parcels.get_fdata() == val).astype(int)

        mask_nii = nib.nifti1.Nifti1Image(
            mask_arr,
            affine=lang_parcels_mni.affine,
            dtype="uint8",
        )
        mask_nii.to_filename(
            Path(
                f"{lang_dir}/binary_masks_mni/"
                f"language-ROIs_Toneva-8_MNI152NLin2009cAsym_{roi_name}.nii.gz"
            )
        )

    """
    Offline: use lang-parcels_mni2T1w.sh to warp individual masks to T1w space
    using ANTS and fmri.prep transformation matrix
    """

else:
    """
    Resample from anat T1w to functional space
    Export as final subject masks at functional resolution
    """
    Path(f"{lang_dir}/binary_masks_t1w").mkdir(parents=True, exist_ok=True)
    m_dir = Path(
        "../../../masks/yeo_networks/"
    ).resolve()

    for snum in ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]:
        func_mask_tw1 = nib.load(f"{m_dir}/binary_masks/{snum}_func-mask_T1w.nii.gz")

        for val, roi_name in roi_dict.items():
            mask_t1w = nib.load(f"{lang_dir}/ROI_parcels_t1w/{snum}_T1w_Language-Toneva8-{roi_name}.nii.gz")

            rs_mask_t1w = resample_to_img(mask_t1w, func_mask_tw1, interpolation="nearest")
            rs_mask_t1w.to_filename(f"{lang_dir}/binary_masks_t1w/{snum}_T1w_Language-Toneva8-{roi_name}.nii.gz")


# TODO: come up w naming convention for masks : parcels, vs binary masks, in anat vs func space...
# TODO: organize output in masks dir

#

# TODO : add masking w grey-matter mask during voxelwise timeseries extraction
# (not needed for parcel-wise, but will reduce number of features for voxelwise target)
# ALso, it's ok to smooth before masking parcels: brings back signal if misalignment
