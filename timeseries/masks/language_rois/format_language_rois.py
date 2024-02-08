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

file_path = Path(
    f"{lang_dir}/from_Toneva/"
    "MNI_language_roi_downsampled_to_mni_t1_2mm_brain_4_8_21.nii.gz"
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

Path(f"{lang_dir}/binary_masks").mkdir(parents=True, exist_ok=True)

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

for val, roi_name in roi_dict.items():
    mask_arr = (lang_parcels.get_fdata() == val).astype(int)

    mask_nii = nib.nifti1.Nifti1Image(
        mask_arr,
        affine=lang_parcels_mni.affine,
        dtype="uint8",
    )
    mask_nii.to_filename(
        Path(f"{lang_dir}/binary_masks/{roi_name}_MNI152NLin2009cAsym.nii.gz")
    )
