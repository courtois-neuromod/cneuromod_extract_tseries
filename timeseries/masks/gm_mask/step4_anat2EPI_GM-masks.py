
import nibabel as nib
from nilearn.image import resample_to_img, get_data, new_img_like, math_img
import numpy as np

mask_path = "/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/masks/gm-masks/freesurfer"
func_path = "/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/masks/yeo-7net/network_masks"


for s in range(1, 7):
    snum = str(s).zfill(2)

    mask_mni_func = nib.load(
        f"{func_path}/sub-{snum}_MNI152NLin2009cAsym_res-"
        "func_desc-func-brain_mask.nii.gz"
    )
    mask_mni_GM = nib.load(
        f"{mask_path}/sub-{snum}_space-MNI152NLin2009cAsym_"
        "label-GM_probseg_FSLinear.nii.gz"
    )

    # export binary GM mask at anatomical resolution w template naming
    nib.save(
        new_img_like(
            mask_mni_GM,
            (mask_mni_GM.get_fdata() > 0.2).astype("uint8"),
        ),
        f"{mask_path}/tpl-MNI152NLin2009cAsym_sub-{snum}"
        "_res-anat_label-GM_desc-from-FS_dseg.nii.gz",
    )

    mask_mni_rs = nib.squeeze_image(
        resample_to_img(
            source_img=mask_mni_GM,
            target_img=mask_mni_func,
            interpolation="linear",
        ),
    )
    mask_mni_tr = new_img_like(
        mask_mni_rs, (mask_mni_rs.get_fdata() > 0.05).astype("uint8")
    )
    #mask_mni_rec = math_img(
    #    "img1 & img2",
    #    img1=mask_mni_func,
    #    img2=mask_mni_tr,
    #)

    nib.save(
        #mask_mni_rec,
        mask_mni_tr,
        f"{mask_path}/tpl-MNI152NLin2009cAsym_sub-{snum}"
        "_res-func_label-GM_desc-from-FS_dseg.nii.gz"
    )


    mask_t1w_func = nib.load(
        f"{func_path}/sub-{snum}_T1w_res-func_desc-"
        "func-brain_mask.nii.gz"
    )
    mask_t1w_GM = nib.load(
        f"{mask_path}/sub-{snum}_space-T1w_label-GM_dseg.nii.gz"
    )
    # export binary GM mask at anatomical resolution w template naming
    nib.save(
        mask_t1w_GM,
        f"{mask_path}/tpl-sub{snum}T1w"
        "_res-anat_label-GM_desc-from-FS_dseg.nii.gz",
    )

    mask_t1w_fl = new_img_like(
        mask_t1w_GM,
        get_data(mask_t1w_GM).astype(np.float64),
    )

    mask_t1w_rs = gm = nib.squeeze_image(
        resample_to_img(
            source_img=mask_t1w_fl,
            target_img=mask_t1w_func,
            interpolation="linear",
        ),
    )

    mask_t1w_tr = new_img_like(
        mask_t1w_rs,
        (get_data(mask_t1w_rs) > 0.2).astype("uint8")
    )

    #mask_t1w_rec = math_img(
    #    "img1 & img2",
    #    img1=mask_t1w_func,
    #    img2=mask_t1w_tr,
    #)

    nib.save(
        #mask_t1w_rec,
        mask_t1w_tr,
        f"{mask_path}/tpl-sub{snum}T1w"
        "_res-func_label-GM_desc-from-FS_dseg.nii.gz"
    )
