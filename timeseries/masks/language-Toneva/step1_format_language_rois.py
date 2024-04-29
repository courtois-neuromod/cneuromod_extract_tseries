"""
Language-relevant ROI masks in MNI space from Mariya Toneva

These ROI were chosen based on previous work showing that they're involved in
language processing, and have been found in several studies to be well
predicted by features from NLP models.
More info here:
https://www.biorxiv.org/content/10.1101/2020.09.28.316935v2)

Step 1: The parcellation atlas is saved as a .nii file with the
.affine matrix that corresponds to functional data in MNI space for
the CNeuroMod subjects.

A binary mask in MNI space (functional resolution) is also created for each
of the 8 ROIs.

Step 2 (step2_lang-parcels_mni2T1w.sh script):
Masks in MNI space are warped to each CNeuromod subject's T1w space (res-anat)
with ANTS using the fmri.prep transformation matrix (script lang-parcels_mni2T1w.sh).

"""

from pathlib import Path
import nibabel as nib
import numpy as np

MNI_FUNC_AFFINE = np.array(
    [[2.0, 0.0, 0.0,  -96.5],
    [ 0.0, 2.0, 0.0, -132.5],
    [ 0.0, 0.0, 2.0,  -78.5],
    [ 0.0, 0.0, 0.0,   1.0]]
)

lang_dir = Path(
    "../../../masks/language-Toneva"
).resolve()

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

Path(f"{lang_dir}/tpl-MNI152NLin2009cAsym").mkdir(parents=True, exist_ok=True)
for i in range(1, 7):
    Path(
        f"{lang_dir}/tpl-sub{str(i).zfill(2)}T1w").mkdir(
            parents=True, exist_ok=True,
    )

file_path = Path(
    f"{lang_dir}/source/"
    "MNI_language_roi_downsampled_to_mni_t1_2mm_brain_4_8_21.nii"
)
lang_parcels = nib.load(file_path)

lang_parcels_mni = nib.nifti1.Nifti1Image(
    lang_parcels.get_fdata().astype(int),
    affine=MNI_FUNC_AFFINE,
    dtype="uint8",
)

lang_parcels_mni.to_filename(
    f"{lang_dir}/tpl-MNI152NLin2009cAsym/"
    "tpl-MNI152NLin2009cAsym_res-func_atlas-langToneva_desc-8_dseg.nii.gz"
)


"""
Generate binary mask in MNI space for each parcel
"""
for val, roi_name in roi_dict.items():
    mask_arr = (lang_parcels.get_fdata() == val).astype(int)

    mask_nii = nib.nifti1.Nifti1Image(
        mask_arr,
        affine=lang_parcels_mni.affine,
        dtype="uint8",
    )
    mask_nii.to_filename(
        Path(
            f"{lang_dir}/tpl-MNI152NLin2009cAsym/"
            f"tpl-MNI152NLin2009cAsym_res-func_atlas-langToneva_label-{roi_name}_mask.nii.gz"
        )
    )
