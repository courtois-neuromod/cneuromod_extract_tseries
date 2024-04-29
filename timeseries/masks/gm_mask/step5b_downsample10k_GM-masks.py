from pathlib import Path

import nibabel as nib
from nilearn.image import resample_img, resample_to_img, get_data, new_img_like
import numpy as np


mask_path = Path(
    "../../../masks/gm-masks/freesurfer/downsampled"
).resolve()
mask_path.mkdir(parents=True, exist_ok=True)

atlas_path = Path(
    "../../../atlases"
).resolve()

target_affine = np.diag((4, 4, 4))

for s in range(1, 7):
    snum = str(s).zfill(2)

    mask_GM_anat = nib.load(
        f"{atlas_path}/tpl-sub{snum}T1w/tpl-sub{snum}T1w_res-anat_label-GM_desc-fromFS_dseg.nii.gz"
    )

    gm_anat_rs_interpC = resample_img(
        mask_GM_anat,
        target_affine=target_affine,
        interpolation='continuous',
    )
    gm_anat_rs_interpC = new_img_like(
        gm_anat_rs_interpC,
        (get_data(gm_anat_rs_interpC) > 0.5).astype("int8"),
    )
    nib.save(
        gm_anat_rs_interpC,
        f"{mask_path}/tpl-sub{snum}T1w_res-04_label-GM_desc-AnatCont_dseg.nii.gz"
    )

    gm_anat_rs_interpNN = resample_img(
        mask_GM_anat,
        target_affine=target_affine,
        interpolation='nearest',
    )
    nib.save(
        gm_anat_rs_interpNN,
        f"{mask_path}/tpl-sub{snum}T1w/tpl-sub{snum}T1w_res-04_label-GM_desc-AnatNN_dseg.nii.gz"
    )


    mask_GM_func = nib.load(
        f"{atlas_path}/tpl-sub{snum}T1w/tpl-sub{snum}T1w_res-func_label-GM_desc-fromFS_dseg.nii.gz"
    )

    gm_func_rs_interpC = resample_img(
        mask_GM_func,
        target_affine=target_affine,
        interpolation='continuous',
    )
    gm_func_rs_interpC = new_img_like(
        gm_func_rs_interpC,
        (get_data(gm_func_rs_interpC) > 0.5).astype("int8"),
    )
    nib.save(
        gm_func_rs_interpC,
        f"{mask_path}/tpl-sub{snum}T1w_res-04_label-GM_desc-FuncCont_dseg.nii.gz"
    )

    gm_func_rs_interpNN = resample_img(
        mask_GM_func,
        target_affine=target_affine,
        interpolation='nearest',
    )
    nib.save(
        gm_func_rs_interpNN,
        f"{mask_path}/tpl-sub{snum}T1w_res-04_label-GM_desc-FuncNN_dseg.nii.gz"
    )
