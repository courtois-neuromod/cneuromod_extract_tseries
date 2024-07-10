import glob, os
from pathlib import Path

import nilearn
from nilearn.image import resample_to_img, new_img_like
import nibabel as nib


mask_path = Path(
    "../../../masks/vision-fLoc/",
).resolve()
func_path = Path(
    "../../../masks/yeo-7net/network_masks"
).resolve()


"""
Resample group-derived parcels from Kanwisher lab from anatomical to
functional resolution (EPI), in MNI space.
"""
mask_mni_func = nib.load(
    f"{func_path}/sub-01_MNI152NLin2009cAsym_res-"
    "func_desc-bold_mask.nii.gz"
)
mask_list = sorted(glob.glob(
    f"{mask_path}/tpl-MNIT1/tpl-MNIT1_res-2mm_atlas-fLocVisionKanwisher_"
    "label-*_pseg.nii",
))

for m in mask_list:
    if "desc-" not in m:
        roi_name = os.path.basename(m).split("_")[-2].split("-")[-1]

        mask_mni_rs = nib.squeeze_image(
            resample_to_img(
                source_img=m,
                target_img=mask_mni_func,
                interpolation="linear",
            ),
        )
        mask_mni_tr = new_img_like(
            mask_mni_rs, (mask_mni_rs.get_fdata() > 0.05).astype("uint8"),
        )

        nib.save(
            mask_mni_tr,
            f"{mask_path}/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_"
            f"res-func_atlas-fLocVisionKanwisher_label-{roi_name}_mask.nii.gz",
        )


"""
For each subject who completed the fLoc task, resample ROI masks
from anatomical to functional resolution (EPI), in MNI space.
"""
for snum in ["01", "02", "03"]:

    mask_mni_func = nib.load(
        f"{func_path}/sub-{snum}_MNI152NLin2009cAsym_res-"
        "func_desc-bold_mask.nii.gz"
    )
    mask_list = sorted(glob.glob(
        f"{mask_path}/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_"
        f"sub-{snum}_res-anat_atlas-fLocVisionTask_label-*_mask.nii.gz",
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
            f"{mask_path}/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_"
            f"sub-{snum}_res-func_atlas-fLocVisionTask_label-{roi_name}_mask.nii.gz",
        )
