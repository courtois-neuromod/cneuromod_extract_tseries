import glob, os
from pathlib import Path

import nibabel as nib
from nilearn.image import resample_to_img, get_data, new_img_like, math_img
import numpy as np

mask_path = Path(
    "../../masks/vision-retino/tpl-MNI152NLin2009cAsym",
).resolve()
func_path = Path(
    "../../masks/yeo-7net/network_masks"
).resolve()


for snum in ["01", "02", "03", "05"]:

    mask_mni_func = nib.load(
        f"{func_path}/sub-{snum}_MNI152NLin2009cAsym_res-"
        "func_desc-bold_mask.nii.gz"
    )
    mask_list = sorted(glob.glob(
        f"{mask_path}/tpl-MNI152NLin2009cAsym_sub-{snum}_res-anat_"
        "atlas-retinoVisionNpythy_label-*_mask.nii.gz",
    ))

    for m in mask_list:
        roi_name = os.path.basename(m).split("_")[-2].split("-")[-1]

        mask_mni_rs = nib.squeeze_image(
            resample_to_img(
                source_img=m,
                target_img=mask_mni_func,
                interpolation="nearest",
            ),
        )

        nib.save(
            mask_mni_rs,
            f"{mask_path}/tpl-MNI152NLin2009cAsym_sub-{snum}_res-func_"
            f"atlas-retinoVisionNpythy_label-{roi_name}_mask.nii.gz",
        )
