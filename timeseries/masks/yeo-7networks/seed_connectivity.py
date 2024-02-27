from typing import List, Tuple, Dict, Union
import glob, os, re
from pathlib import Path
import argparse

import numpy as np
import nibabel as nib
from nibabel import Nifti1Image
import pandas as pd
from tqdm import tqdm
import nilearn.interfaces
from nilearn.image import mean_img, resample_to_img
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker, NiftiSpheresMasker
from nilearn.masking import compute_multi_epi_mask
from scipy.stats import zscore


SEEDS = (
    ("visual", (-16, -74, 7), 1, 0.085),
    ("sensorimotor", (-41, -20, 62), 2, 0.06),
    ("dorsal-attention", (-34, -38, 44), 3, 0.075),
    ("ventral-attention", (-5, 15, 32), 4, 0.05),  # (-31, 11, 8)),
    ("fronto-parietal", (-40, 50, 7), 5, 0.057),
    ("default-mode", (-7, -52, 26), 6, 0.12),
)

LOAD_CONFOUNDS_PARAMS = {
    "strategy": ["motion", "high_pass", "wm_csf", "global_signal"],
    "motion": "basic",
    "wm_csf": "basic",
    "global_signal": "basic",
    "demean": True,
}


def get_lists(
    data_dir: str,
    found_masks: List[str],
    space_name: str,
) -> Tuple[List[str], List[str]]:
    """."""
    bold_list = []
    mask_list = []
    for fm in found_masks:
        identifier = fm.split('/')[-1].split('_space')[0]
        sub, ses = identifier.split('_')[:2]

        bpath = sorted(glob.glob(
            f"{data_dir}/{sub}/{ses}/func/{identifier}"
            f"*_space-{space_name}*_desc-preproc_*bold.nii.gz"
        ))

        if len(bpath) == 1 and Path(bpath[0]).exists():
            bold_list.append(bpath[0])
            mask_list.append(fm)

    return bold_list, mask_list


def get_dset_func_mask(
    mask_list: List[str],
) -> Nifti1Image :
    """."""
    return compute_multi_epi_mask(
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


def get_GM_mask(
    args: argparse.Namespace,
    func_mask: Nifti1Image,
    space_name: str,
) -> Nifti1Image:
    """."""
    if space_name=="T1w":
        gm_mask = nib.load(
            f"{args.atlas_dir}/tpl-sub{args.subject}T1w/tpl-sub"
            f"{args.subject}T1w_res-func_label-GM_desc-from-FS_dseg.nii.gz"
        )
    else:
        gm_mask = nib.load(
            f"{args.atlas_dir}/tpl-MNI152NLin2009cAsym/"
            f"tpl-MNI152NLin2009cAsym_sub-{args.subject}_res-func_"
            "label-GM_desc-from-FS_dseg.nii.gz"
        )

    return nib.nifti1.Nifti1Image(
        (gm_mask.get_fdata()*func_mask.get_fdata()).astype(int),
        affine=func_mask.affine,
        dtype="uint8",
    )


def get_fmri_files(
    args: argparse.Namespace,
    space_name: str,
) -> Tuple[List[str], List[pd.DataFrame], Nifti1Image, Nifti1Image]:
    """."""
    found_masks = sorted(
        glob.glob(
            f"{args.data_dir}/sub-{args.subject}/"
            f"ses-*/func/*task-restingstate*_space-{space_name}*_mask.nii.gz",
    ))

    bold_list, mask_list = get_lists(args.data_dir, found_masks, space_name)
    dset_func_mask = get_dset_func_mask(mask_list)
    gm_mask = get_GM_mask(args, dset_func_mask, space_name)

    confounds, _ = nilearn.interfaces.fmriprep.load_confounds(
        bold_list,
        **LOAD_CONFOUNDS_PARAMS,
    )

    return bold_list, confounds, dset_func_mask, gm_mask


def generate_mni_parcel_masks(
    args: argparse.Namespace,
) -> None:
    """
    Generates mni parcel masks (using MIST-ROI parcellation) from the
    network's seed coordinate (Yeo 2011 7-networks) if mask doesn't already
    exist.
    """
    Path(f"{args.out_dir}/parcel_masks").mkdir(parents=True, exist_ok=True)
    Path(f"{args.out_dir}/network_masks").mkdir(parents=True, exist_ok=True)
    Path(f"{args.out_dir}/temp").mkdir(parents=True, exist_ok=True)

    atlas_parcel = nib.load(Path(
        f"{args.atlas_dir}/{args.atlas_space}/{args.atlas_name}"
    ).resolve())
    for seed in SEEDS:
        mni_parcel_mask_path = Path(
            f"{args.out_dir}/parcel_masks/{args.atlas_space}"
            f"_{args.atlas_res}_atlas-yeo-7net_desc-{seed[0]}-seed-parcel_mask.nii.gz"
        )
        if not mni_parcel_mask_path.exists():
            mni_seed_masker = NiftiSpheresMasker(
                [seed[1]],
                radius=1,
                detrend=False,
                standardize=False,
                verbose=0,
            )
            ROI_val = mni_seed_masker.fit_transform(atlas_parcel)[0][0]

            parcel_mask = nib.nifti1.Nifti1Image(
                (atlas_parcel.get_fdata() == ROI_val).astype(int),
                affine=atlas_parcel.affine,
                dtype="uint8",
            )
            assert np.sum(parcel_mask.get_fdata()) > 0.0

            nib.save(parcel_mask, mni_parcel_mask_path)


def make_mask_array(
    args: argparse.Namespace,
    parcel: dict,
    n_ROI: int,
    GM_mask: Nifti1Image,
    space: str,
) -> np.array:
    """ . """
    if space == "MNI":
        parcel_mask_path = Path(
            f"{args.out_dir}/parcel_masks/"
            f"{args.atlas_space}_{args.atlas_res}_atlas-yeo-7net_desc-"
            f"{parcel['network_name']}-seed-parcel_mask.nii.gz"
        )
    else:
        parcel_mask_path = Path(
            f"{args.out_dir}/parcel_masks/"
            f"tpl-sub{args.subject}T1w_res-anat_atlas-yeo-7net_desc-"
            f"{parcel['network_name']}-seed-parcel_mask.nii.gz"
        )
    parcel_mask = nib.load(parcel_mask_path)
    parcel_mask = resample_to_img(
        parcel_mask, GM_mask, interpolation="nearest",
    )

    return parcel_mask.get_fdata()*GM_mask.get_fdata()*n_ROI


def make_labels_parcel_masks(
    args: argparse.Namespace,
    mni_GM_mask: Nifti1Image,
    t1w_GM_mask: Nifti1Image,

) -> Tuple[Dict, Nifti1Image, Nifti1Image]:
    """ . """
    parcels_ROI = {s[2]:{"network_name": s[0]} for s in SEEDS}

    for n_ROI, parcel in parcels_ROI.items():
        parcels_ROI[n_ROI]["mni_mask"] = make_mask_array(
            args, parcel, n_ROI, mni_GM_mask, "MNI",
        )
        parcels_ROI[n_ROI]["t1w_mask"] = make_mask_array(
            args, parcel, n_ROI, t1w_GM_mask, "T1w",
        )

    mni_labels_mask = nib.nifti1.Nifti1Image(
        np.sum(np.stack(
            [x["mni_mask"] for x in parcels_ROI.values()], axis=-1
        ), axis=-1),
        affine=mni_GM_mask.affine,
        dtype="uint8",
    )

    t1w_labels_mask = nib.nifti1.Nifti1Image(
        np.sum(np.stack(
            [x["t1w_mask"] for x in parcels_ROI.values()], axis=-1
        ), axis=-1),
        affine=t1w_GM_mask.affine,
        dtype="uint8",
    )

    return parcels_ROI, mni_labels_mask, t1w_labels_mask


def compute_connectivity(
    args: argparse.Namespace,
    parcels_ROI: Dict,
    bold_list: List[str],
    func_mask: Nifti1Image,
    labels_mask: Nifti1Image,
    confounds: List[pd.DataFrame],
) -> Dict:
    """."""

    connectivity_lists = {s[0]:[] for s in SEEDS}
    for i, file_path in tqdm(enumerate(bold_list)):
        filename = os.path.split(file_path)[1].split("_desc-")[0]

        brain_masker = NiftiMasker(
            mask_img=func_mask,
            detrend=False,
            standardize="zscore_sample",
            smoothing_fwhm=5,
        )
        brain_timeseries = brain_masker.fit_transform(
            file_path,
            confounds=confounds[i],
        )
        denoised_brain = brain_masker.inverse_transform(
            brain_timeseries,
        )

        parcel_masker = NiftiLabelsMasker(
            labels_img=labels_mask,
            detrend=False,
            standardize=False,
        )
        parcel_timeseries = parcel_masker.fit_transform(
            denoised_brain,
        )

        for i, r_id in enumerate(parcel_masker.labels_):
            net_name = parcels_ROI[int(r_id)]['network_name']
            parcel_corr = (
                np.dot(brain_timeseries.T, parcel_timeseries[:, i])
                / parcel_timeseries.shape[0]
            )

            out_path = f"{args.out_dir}/temp/{filename}_desc-{net_name}_connectivity.nii.gz"
            connectivity_lists[net_name].append(out_path)
            brain_masker.inverse_transform(parcel_corr.T).to_filename(out_path)

    return connectivity_lists


def create_network_masks(
    args: argparse.Namespace,
    space: str,
    fconnect_lists: Dict,
    func_mask: Nifti1Image,
    GM_mask: Nifti1Image,
) -> None:
    """."""
    s_name = f"tpl-sub{args.subject}T1w" if space == "T1w" else f"tpl-MNI152NLin2009cAsym_sub-{args.subject}"

    func_mask.to_filename(
        f"{args.out_dir}/network_masks/"
        f"sub-{args.subject}_{space}_res-func_desc-func-brain_mask.nii.gz"
    )
    GM_mask.to_filename(
        f"{args.out_dir}/network_masks/"
        f"sub-{args.subject}_{space}_res-func_desc-GM_mask.nii.gz"
    )

    for seed in SEEDS:
        mean_connectivity = mean_img(fconnect_lists[seed[0]])
        mean_connectivity.to_filename(
            f"{args.out_dir}/network_masks/"
            f"sub-{args.subject}_{space}_res-func_atlas-yeo-7net_"
            f"desc-{seed[0]}_avg-connectivity.nii.gz"
        )

        z_masker = NiftiMasker(
            mask_img=func_mask,
            detrend=False,
            standardize=False,
            smoothing_fwhm=None,
        )
        z_connectivity = np.squeeze(z_masker.fit_transform(mean_connectivity))
        network_mask = z_masker.inverse_transform(
            (zscore(z_connectivity, nan_policy='propagate') > 2.0).astype("int32"),
        )

        #nvox = int(np.sum(func_mask.get_fdata())*seed[3])
        #vox_cutoff = np.sort(mean_connectivity.get_fdata().reshape([-1]))[-nvox]
        #network_arr = (mean_connectivity.get_fdata() >= vox_cutoff).astype(int)
        #network_mask = nib.nifti1.Nifti1Image(
        #    network_arr,
        #    affine=mean_connectivity.affine,
        #    dtype="uint8",
        #)

        network_mask.to_filename(
            f"{args.out_dir}/network_masks/"
            f"{s_name}_res-func_atlas-yeo-7net_desc-{seed[0]}_mask.nii.gz"
        )
        network_mask_GM = nib.nifti1.Nifti1Image(
            (network_mask.get_fdata()*GM_mask.get_fdata()).astype(int),
            affine=GM_mask.affine,
            dtype="uint8",
        )
        network_mask_GM.to_filename(
            f"{args.out_dir}/network_masks/"
            f"{s_name}_res-func_atlas-yeo-7net_desc-{seed[0]}-GM-masked_mask.nii.gz"
        )


def main(args: argparse.Namespace):
    """
    Derives network masks for 6 of the 7 Yeo 2011 networks in MNI and T1w space
    for each CNeuroMod subject.

    For each network, for each run, extracts time-series from the MIST-ROI
    parcel that includes a seed voxel (MNI coordinates from Yeo 2011),
    and correlates its signal with all brain voxels within a grey matter
    mask.

    Voxel-wise correlations are averaged across all resting-state runs, and a
    percentage of voxels with the highest R scores is select to form a
    binary mask for each network.
    Network-specific percentages are proportional to the number of voxels
    within each network in the Yeo 2011 7-network parcellation.

    Script adapted from
    https://github.com/courtois-neuromod/fmri-generator/blob/master/scripts/seed_connectivity.py

    Based on nilearn tutorial
    https://nilearn.github.io/stable/auto_examples/03_connectivity/plot_seed_to_voxel_correlation.html

    Yeo et al., 2011 for coordinates
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3174820/
    """
    print(args)

    generate_mni_parcel_masks(args)

    found_t1w_parcel_masks = np.sum([
        Path(
            f"{args.out_dir}/parcel_masks/tpl-sub{args.subject}T1w_res-anat"
            f"_atlas-yeo-7net_desc-{seed[0]}-seed-parcel_mask.nii.gz"
        ).exists() for seed in SEEDS]) == len(SEEDS)

    if not found_t1w_parcel_masks:
        """
        OFFLINE work: generate T1w parcel masks
        For each subject, convert each PARCEL mask from MNI to T1w space w ANTS
        using yeo-parcel_mni2T1w.sh script.
        Save as
        <args.out_dir>/parcel_masks/tpl-sub<args.subject>T1w_res-anat_atlas-yeo-7net_desc-<seed[0]>-seed-parcel_mask.nii.gz
        """
        print(
            "Missing T1w parcel masks. Use ants to generate parcel masks in"
            " subject's T1w space."
        )

    else:
        """
        Generates network masks only if all seed masks exist (MNI and T1w)
        """
        (
            mni_bold_list, mni_confounds, mni_func_mask, mni_GM_mask,
        ) = get_fmri_files(
            args, "MNI152NLin2009cAsym")

        (
            t1w_bold_list, t1w_confounds, t1w_func_mask, t1w_GM_mask,
        ) = get_fmri_files(args, "T1w")

        (
            parcels_ROI,
            mni_labels_mask,
            t1w_labels_mask,
        ) = make_labels_parcel_masks(
            args,
            mni_GM_mask,
            t1w_GM_mask,
        )

        print("Computing functional connectivity in MNI space")
        mni_fconnect_lists = compute_connectivity(
            args,
            parcels_ROI,
            mni_bold_list,
            mni_func_mask,
            mni_labels_mask,
            mni_confounds,
        )
        print("Creating network masks in MNI space")
        create_network_masks(
            args,
            "MNI152NLin2009cAsym",
            mni_fconnect_lists,
            mni_func_mask,
            mni_GM_mask,
        )

        print("Computing functional connectivity in T1w space")
        t1w_fconnect_lists = compute_connectivity(
            args,
            parcels_ROI,
            t1w_bold_list,
            t1w_func_mask,
            t1w_labels_mask,
            t1w_confounds,
        )
        print("Creating network masks in T1w space")
        create_network_masks(
            args,
            "T1w",
            t1w_fconnect_lists,
            t1w_func_mask,
            t1w_GM_mask,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output directory.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Base directory for data files.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        help="Subject to use (e.g. '01').",
    )
    parser.add_argument(
        "--atlas_dir",
        type=str,
        default="../../../atlases",
    )
    parser.add_argument(
        "--atlas_space",
        type=str,
        default="tpl-MNI152NLin2009bSym",
    )
    parser.add_argument(
        "--atlas_res",
        type=str,
        default="res-03",
    )    
    parser.add_argument(
        "--atlas_name",
        type=str,
        default="tpl-MNI152NLin2009bSym_res-03_atlas-MIST_desc-ROI_dseg.nii.gz",
    )

    main(parser.parse_args())
