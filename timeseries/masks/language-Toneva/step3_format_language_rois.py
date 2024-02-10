"""
Language-relevant ROI masks in MNI space from Mariya Toneva

These ROI were chosen based on previous work showing that they're involved in
language processing, and have been found in several studies to be well
predicted by features from NLP models.
More info here:
https://www.biorxiv.org/content/10.1101/2020.09.28.316935v2)

Step 3: binary masks in T1w space are resampled from anat to
functional (BOLD) resolution (saved as final binary masks)

"""
from pathlib import Path
import nibabel as nib
from nilearn.image import resample_to_img
import numpy as np


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

# TODO : update to permanent file...
m_dir = Path(
    "../../../masks/yeo_networks/"
).resolve()

for snum in ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]:
    func_mask_tw1 = nib.load(f"{m_dir}/binary_masks/{snum}_func-mask_T1w.nii.gz")

    for val, roi_name in roi_dict.items():
        mask_t1w = nib.load(
            f"{lang_dir}/tpl-sub{snum}T1w/tpl-sub{snum}T1w_res-anat_"
            f"atlas-language-Toneva_desc-{roi_name}_mask.nii.gz"
        )

        rs_mask_t1w = resample_to_img(mask_t1w, func_mask_tw1, interpolation="nearest")
        rs_mask_t1w.to_filename(
            f"{lang_dir}/tpl-sub{snum}T1w/tpl-sub{snum}T1w_res-func_"
            f"atlas-language-Toneva_desc-{roi_name}_mask.nii.gz"
        )
