import glob
from pathlib import Path

import nilearn
from nilearn.masking import compute_multi_epi_mask
import nibabel as nib


"""
For each subject, generate a grey matter mask from the friends dataset

Resample the Kanwisher parcel-derived mask to the EPI mask, then threshold
the parcel to create a binary mask
"""

for sub in range(1, 7):
    sub_num = str(sub).zfill(2)

    #TODO: make greymatter mask
    found_mask_list = sorted(
        glob.glob(
            "/home/mstlaure/projects/rrg-pbellec/mstlaure/"
            "cneuromod_extract_tseries/data/friends.fmriprep"
            f"/sub-{sub_num}/ses-*/func/*T1w*_mask.nii.gz",
            ),
        )

    bold_list = []
    mask_list = []
    for fm in found_mask_list:
        identifier = fm.split('/')[-1].split('_space')[0]
        s, ses = identifier.split('_')[:2]

        bpath = sorted(glob.glob(
            "/home/mstlaure/projects/rrg-pbellec/mstlaure/"
            "cneuromod_extract_tseries/data/friends.fmriprep"
            f"/{s}/{ses}/func/{identifier}"
            f"*T1w*_desc-preproc_*bold.nii.gz"
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

    parcel_list = glob.glob(
        "/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/"
        "masks/standard_masks/fLoc_kanwisher_parcels/ROI_parcels_t1w/"
        f"sub-{sub_num}_kanwisher*.nii"
    )

    for parcel_path in parcel_list:
        pname = parcel_path.split('/')[-1].split("_")[1:3]
        parcel = nib.load(parcel_path)

        rs_parcel = nilearn.image.resample_to_img(parcel, subject_epi_mask, interpolation='continuous')
        rs_parcel = nib.nifti1.Nifti1Image((rs_parcel.get_fdata() > 0.5).astype(int), affine=rs_parcel.affine, dtype="uint8")

        mpath = Path(
            "/home/mstlaure/projects/rrg-pbellec/mstlaure/"
            "cneuromod_extract_tseries/masks/subject_masks/fLoc/"
            f"sub-{sub_num}/sub-{sub_num}_T1w_{pname[0]}_{pname[1].split('.')[0]}.nii"
        )
        nib.save(rs_parcel, mpath)
