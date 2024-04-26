import glob
from pathlib import Path
import argparse

import nibabel as nib
import nilearn
import nilearn.interfaces
from nilearn.image import resample_img, resample_to_img, get_data, new_img_like
from nilearn.maskers import NiftiMasker
from nilearn.masking import compute_multi_epi_mask
from nilearn.regions import Parcellations
import numpy as np
import pandas as pd
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    "--subject",
    type=str,
    help="Subject to use (e.g. '01').",
)
args = parser.parse_args()

LOAD_CONFOUNDS_PARAMS = {
    "strategy": ["motion", "high_pass", "wm_csf", "global_signal"],
    "motion": "basic",
    "wm_csf": "basic",
    "global_signal": "basic",
    "demean": True,
}
bold_path = Path(
    "../../../../friends_algonauts/data/friends.fmriprep2"
).resolve()
atlas_path = Path("../../../atlases").resolve()
out_path = Path("../../../masks/gm-masks/parcellation").resolve()

parcellation = nib.load(
    f"{atlas_path}/tpl-MNI152NLin2009cAsym/"
    "tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_"
    "desc-1000Parcels7Networks_dseg.nii.gz"
)

snum = args.subject
gm_mask_path = Path(
    f"{out_path}/tpl-MNI152NLin2009cAsym_sub-{snum}_task-friends_season-06_"
    "label-bold_mask.nii.gz"
)

found_masks = sorted(
    glob.glob(
        f"{bold_path}/sub-{snum}/"
        "ses-*/func/*task-s06e*space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
))

bold_list = []
mask_list = []
for fm in found_masks:
    identifier = fm.split('/')[-1].split('_space')[0]
    sub, ses = identifier.split('_')[:2]

    bpath = sorted(glob.glob(
        f"{bold_path}/{sub}/{ses}/func/{identifier}"
        "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    ))

    if len(bpath) == 1 and Path(bpath[0]).exists():
        bold_list.append(bpath[0])
        mask_list.append(fm)

dset_func_mask = compute_multi_epi_mask(
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
dset_func_mask.to_filename(gm_mask_path)


# already matching...
parcellation_rs = resample_to_img(
    parcellation, dset_func_mask, interpolation="nearest",
)
roi_list = [x for x in np.unique(parcellation_rs.get_fdata()) if x > 0.0]

gm_mask = nib.load(
    f"{atlas_path}/tpl-MNI152NLin2009cAsym/"
    f"tpl-MNI152NLin2009cAsym_sub-{snum}_res-func_"
    "label-GM_desc-fromFS_dseg.nii.gz"
)
gm_func_mask = nib.nifti1.Nifti1Image(
    (gm_mask.get_fdata()*dset_func_mask.get_fdata()).astype(int),
    affine=dset_func_mask.affine,
    dtype="uint8",
)


# Ward parcellation without clusters...
wpath = Path(
    f"{out_path}/tpl-MNI152NLin2009cAsym_sub-{snum}_res-func_"
    "atlas-10kparcels_desc-Ward_dseg.nii.gz"
)
if not wpath.exists():
    confounds, _ = nilearn.interfaces.fmriprep.load_confounds(
        bold_list,
        **LOAD_CONFOUNDS_PARAMS,
    )
    denoised_bold_list = []
    for i, file_path in tqdm(enumerate(bold_list)):
        identifier = file_path.split('/')[-1].split('_desc')[0]
        fpath = Path(
            f"{out_path}/temp/{identifier}_desc-denoised_bold.nii.gz"
        )
        if not fpath.exists():
            brain_masker = NiftiMasker(
                mask_img=gm_func_mask,
                detrend=False,
                standardize="zscore_sample",
                smoothing_fwhm=8,
            )
            brain_timeseries = brain_masker.fit_transform(
                file_path,
                confounds=confounds[i],
            )
            denoised_brain = brain_masker.inverse_transform(
                brain_timeseries,
            )
            denoised_brain.to_filename(fpath)
        denoised_bold_list.append(fpath)

    ward = Parcellations(
        method="ward",
        n_parcels=10000,
        mask=gm_func_mask,
        standardize=False,
        smoothing_fwhm=None,
        detrend=False,
        verbose=1,
    )
    ward.fit(denoised_bold_list)
    ward_labels_img = ward.labels_img_
    ward_labels_img.to_filename(wpath)
else:
    ward_labels_img = nib.load(wpath)


# Extract descriptors, assign 10k parcels to Schaefer18_1000Parcels7Networks
flat_parcellation = parcellation_rs.get_fdata().reshape((-1))

roi_list = [x for x in np.unique(ward_labels_img.get_fdata()) if x > 0.0]
parcel_match = []
for r in sorted(roi_list):
    roi_mask = (ward_labels_img.get_fdata() == r).reshape((-1)).astype(int)
    vals, counts = np.unique(roi_mask*flat_parcellation, return_counts=True)

    # pick value with 2nd highest vox count (highest count is 0 = outside mask)
    s_id = vals[np.argpartition(-counts, kth=1)[1]] if len(vals) > 1 else np.nan
    parcel_match.append([r, np.sum(roi_mask), s_id])

idx_df = pd.DataFrame(
    np.array(parcel_match),
    columns=["10kParcel", "vox_count", "schaefer18_1kParcel7Net"],
)
idx_df.to_csv(
    f"{out_path}/tpl-MNI152NLin2009cAsym_sub-{snum}_atlas-10kparcels_desc-Ward_dseg.tsv",
    sep = "\t", header=True, index=False,
)


"""
# parcellation within shaefer-1000 clusters: 10 subclusters per cluster
subparcel_path = Path(
    f"{out_path}/tpl-MNI152NLin2009cAsym_sub-{snum}_res-func_"
    "atlas-Schaefer2018_desc-10kSubparcelsReNa_dseg.nii.gz"
)
if not subparcel_path.exists():
    cluster_label_list = []
    for r in tqdm(roi_list):
        roi_mask = nib.nifti1.Nifti1Image(
            (parcellation_rs.get_fdata() == r).astype(int),
            affine=parcellation_rs.affine,
            dtype="uint8",
        )
        rena = Parcellations(
            method="rena",
            n_parcels=10,
            mask=roi_mask,
            standardize=False,
            smoothing_fwhm=None,
            detrend=False,
            verbose=1,
        )

        rena.fit(denoised_bold_list)
        rena_labels_img = rena.labels_img_
        cluster_label_list.append(
            roi_mask.get_fdata()*((r*100) + rena_labels_img.get_fdata())
        )

    final_parcellation = nib.nifti1.Nifti1Image(
        np.sum(cluster_label_list, axis=0).astype(int),
        affine=parcellation_rs.affine,
        dtype="uint8",
    )
    final_parcellation.to_filename(subparcel_path)



# Recursive binary parcellation within Schaefer1000...
for r in roi_list:
    out_file = Path(
        f"{out_path}/temp/tpl-MNI152NLin2009cAsym_sub-{snum}_res-func_"
        f"atlas-Schaefer2018_desc-10kSubparcelsReNa_parcel-{int(r)}_dseg.npy"
    )
    if not out_file.exists():
        roi_mask = nib.nifti1.Nifti1Image(
            (parcellation_rs.get_fdata() == r).astype(int),
            affine=parcellation_rs.affine,
            dtype="uint8",
        )
        pcount = 0
        rena_labels, pcount = recursive_clustering(
            r,
            pcount,
            np.zeros(roi_mask.get_fdata().shape),#.astype(int),
            roi_mask,
            denoised_bold_list,
        )
        np.save(out_file, rena_labels)

    cluster_label_list = [np.load(x) for x in sorted(glob.glob(
        f"{out_path}/temp/tpl-MNI152NLin2009cAsym_sub-{snum}_res-func_"
        "atlas-Schaefer2018_desc-10kSubparcelsReNa_parcel-*_dseg.npy"
    ))]

    final_parcellation = nib.nifti1.Nifti1Image(
        np.sum(cluster_label_list, axis=0).astype(int),
        affine=parcellation_rs.affine,
        dtype="uint8",
    )

    final_parcellation.to_filename(
        f"{out_path}/tpl-MNI152NLin2009cAsym_sub-{snum}_res-func_"
        "atlas-Schaefer2018_desc-10kSubparcelsReNaRecursive_dseg.nii.gz"
    )


def recursive_clustering(
    r,
    pcount,
    cummul_labels,
    roi_mask,
    denoised_bold_list,
):
    rena = Parcellations(
        method="rena",
        n_parcels=2,
        mask=roi_mask,
        standardize=False,
        smoothing_fwhm=None,
        detrend=False,
        verbose=1,
    )
    rena.fit(denoised_bold_list)
    rena_labels_img = rena.labels_img_

    idx, counts = np.unique(rena_labels_img.get_fdata(), return_counts=True)
    assert len(idx ==3)   # vals should be [0.0, 1.0 and 2.0]

    r_idx = [(idx[x], counts[x]) for x in range(3) if idx[x] > 0]
    for id, c in r_idx:
        if c > 2:
            roi_submask = nib.nifti1.Nifti1Image(
                (rena_labels_img.get_fdata() == id).astype(int),
                affine=roi_mask.affine,
                dtype="uint8",
            )
            if c < 25:
                pcount += 1
                cummul_labels += (roi_submask.get_fdata()*((r*100) + pcount))
            else:
                cummul_labels, pcount = recursive_clustering(
                    r, pcount, cummul_labels, roi_submask, denoised_bold_list,
                )

    return cummul_labels, pcount


def recursive_clustering_old(
    r,
    pcount,
    cummul_labels,
    roi_mask,
    denoised_bold_list,
):
    if np.sum(roi_mask.get_fdata()) < 11:
        pcount += 1
        cummul_labels += (roi_mask.get_fdata()*((r*100) + pcount))
        return cummul_labels, pcount
    else:
        rena = Parcellations(
            method="rena",
            n_parcels=2,
            mask=roi_mask,
            standardize=False,
            smoothing_fwhm=None,
            detrend=False,
            verbose=1,
        )
        rena.fit(denoised_bold_list)
        rena_labels_img = rena.labels_img_

        idx, counts = np.unique(rena_labels_img.get_fdata(), return_counts=True)
        assert len(idx ==3)   # vals should be [0.0, 1.0 and 2.0]
        if np.sum([x < 11 for x in counts]) > 0:
            # Cluster too small, refuse parcellation
            pcount += 1
            cummul_labels += (roi_mask.get_fdata()*((r*100) + pcount))
            return cummul_labels, pcount

        else:
            r_idx = [x for x in idx if x > 0]
            for id in r_idx:
                roi_submask = nib.nifti1.Nifti1Image(
                    (rena_labels_img.get_fdata() == id).astype(int),
                    affine=roi_mask.affine,
                    dtype="uint8",
                )
                cummul_labels, pcount = recursive_clustering_old(
                    r, pcount, cummul_labels, roi_submask, denoised_bold_list,
                )
            return cummul_labels, pcount
"""
