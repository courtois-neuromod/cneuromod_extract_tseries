from typing import List, Tuple, Dict, Union
import glob, os, re
from pathlib import Path
import argparse

import numpy as np
import nibabel as nib
from nibabel import Nifti1Image
import pandas as pd
import pickle as pk
from tqdm import tqdm
import nilearn.interfaces
from nilearn.image import mean_img
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker, NiftiSpheresMasker
from nilearn.masking import compute_multi_epi_mask


SEEDS = (
    ("visual", (-16, -74, 7), 1),
    ("sensorimotor", (-41, -20, 62), 2),
    ("dorsal_attention", (-34, -38, 44), 3),
    ("ventral_attention", (-5, 15, 32), 4),  # (-31, 11, 8)),
    ("fronto-parietal", (-40, 50, 7), 5),
    ("default-mode", (-7, -52, 26), 6),
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


def get_dset_mask(
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


def get_fmri_files(
    args: argparse.Namespace,
    space_name: str,
) -> Tuple[List[str], List[pd.DataFrame], Nifti1Image]:
    """."""
    found_masks = sorted(
        glob.glob(
            f"{args.data_dir}/{args.subject}/"
            f"ses-*/func/*_space-{space_name}*_mask.nii.gz",
    ))
    found_masks = [p for p in found_masks if re.search(args.task_filter, p)]

    bold_list, mask_list = get_lists(args.data_dir, found_masks, space_name)
    dset_GM_mask = get_dset_mask(mask_list)

    confounds, _ = nilearn.interfaces.fmriprep.load_confounds(
        bold_list,
        **LOAD_CONFOUNDS_PARAMS,
    )

    return bold_list, confounds, dset_GM_mask


def generate_mni_seed_masks(
    args: argparse.Namespace,
    mni_GM_mask: Nifti1Image,
) -> None:
    """ . """
    for seed in SEEDS:
        mni_seed_mask_path = Path(
            f"{args.out_dir}/{args.subject}_{seed[0]}_seed_MNI.nii.gz"
        )
        if not mni_seed_mask_path.exists():
            mni_seed_masker = NiftiSpheresMasker(
                [seed[1]],
                radius=1,
                mask_img=mni_GM_mask,
                detrend=False,
                standardize=False,
                verbose=0,
            )

            mni_seed_vox = mni_seed_masker.fit_transform(mni_GM_mask)
            assert mni_seed_vox[0][0] == 1.0
            mni_seed_mask = mni_seed_masker.inverse_transform(mni_seed_vox)

            nib.save(mni_seed_mask, mni_seed_mask_path)


def make_labels_seed_masks(
    args: argparse.Namespace,
    mni_GM_mask: Nifti1Image,
    t1w_GM_mask: Nifti1Image,

) -> Tuple[Dict, Nifti1Image, Nifti1Image]:
    """ . """
    seeds_ROI = {s[2]:{"ROI_name": s[0]} for s in SEEDS}

    for n_ROI, seed in seeds_ROI.items():
        mni_seed_mask_path = Path(
            f"{args.out_dir}/{args.subject}_{seed}_seed_MNI.nii.gz"
        )
        mni_seed_mask = nib.load(mni_seed_mask_path)
        seeds_ROI[n_ROI]["mni_mask"] = mni_seed_mask.get_fdata()*n_ROI

        t1w_seed_mask_path = Path(
            f"{args.out_dir}/{args.subject}_{seed}_seed_T1w.nii.gz"
        )
        t1w_seed_mask = ni.load(t1w_seed_mask_path)
        t1w_seed_mask = resample_to_img(
            t1w_seed_mask, t1w_GM_mask, interpolation="nearest"
        )
        t1w_seed_max = np.max(t1w_seed_mask.get_fdata())
        seeds_ROI[n_ROI]["t1w_mask"] = (
            t1w_seed_max.get_fdata() == t1w_seed_max
        ).astype(int)*n_ROI

    # debugging sanity check; remove later
    mlabels = np.sum([x["mni_mask"] for x in seeds_ROI.values()])
    tlabels = np.sum([x["t1w_mask"] for x in seeds_ROI.values()])
    assert len(mlabels.shape) == 3
    assert np.sum(mlabels > 0) == len(SEEDS)
    assert len(tlabels.shape) == 3
    assert np.sum(tlabels > 0) == len(SEEDS)

    mni_labels_mask = nib.nifti1.Nifti1Image(
        np.sum([x["mni_mask"] for x in seeds_ROI.values()]),
        affine=mni_GM_mask.affine,
        dtype="uint8",
    )

    t1w_labels_mask = nib.nifti1.Nifti1Image(
        np.sum([x["t1w_mask"] for x in seeds_ROI.values()]),
        affine=t1w_GM_mask.affine,
        dtype="uint8",
    )

    return seeds_ROI, mni_labels_mask, t1w_labels_mask


def compute_connectivity(
    args: argparse.Namespace,
    space: str,
    seeds_ROI: Dict,
    bold_list: List[str],
    GM_mask: Nifti1Image,
    labels_mask: Nifti1Image,
    confounds: List[pd.DataFrame],
) -> Dict:
    """."""

    connectivity_lists = {}
    for i, file_path in tqdm(enumerate(bold_list)):
        filename = os.path.split(file_path)[1].replace(".nii.gz", "")

        brain_masker = NiftiMasker(
            mask_img=GM_mask,
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

        seed_masker = NiftiLabelsMasker(
            labels_img=labels_mask,
            detrend=False,
            standardize=False,
        )
        seed_timeseries = seed_masker.fit_transform(
            denoised_brain,
        )

        for i, r_id in seed_masker.region_ids_.items():
            seed_name = seeds_ROI[int(r_id)]
            seed_vox_corr = (
                np.dot(brain_time_series.T, seed_time_series[:, i])
                / seed_time_series.shape[0]
            )

            out_path = f"{args.out_dir}/temp/{filename}_{seed_name}_{space}_connectivity.nii.gz"
            connectivity_lists[seed_name].append(out_path)
            brain_masker.inverse_transform(seed_vox_corr.T).to_filename(out_path)

    return connectivity_lists


def create_network_masks(
    args: argparse.Namespace,
    space: str,
    fconnect_lists: Dict,
    GM_mask: Nifti1Image,
) -> None:
    """."""

    nvox = int(np.sum(GM_mask)*0.1)  # add parameters?
    for seed in SEEDS:
        mean_connectivity = mean_img(fconnect_lists[seed[0]])

        vox_cutoff = np.sort(mean_connectivity)[-nvox]
        network_mask = nib.nifti1.Nifti1Image(
            (mean_connectivity.get_fdata() >= vox_cutoff).astype(int),
            affine=mean_connectivity.affine,
        )
        network_mask.to_filename(
            f"{args.out_dir}/{args.subject}_{seed[0]}_{space}_mask.nii.gz"
        )


def main(args: argparse.Namespace):
    """
    Analysis performed in MNI and T1w space.

    For each network, for each run, extract time-series from seed voxel and
    correlate with signal from all other brain voxels within an
    average grey matter mask.

    Average the voxel-wise correlations across all task runs, and select
    voxels with the 10% highest r scores within the dataset's grey matter mask.

    Script adapted from
    https://github.com/courtois-neuromod/fmri-generator/blob/master/scripts/seed_connectivity.py

    Based on nilearn tutorial
    https://nilearn.github.io/stable/auto_examples/03_connectivity/plot_seed_to_voxel_correlation.html
    """

    mni_bold_list, mni_confounds, mni_GM_mask = get_fmri_files(
        args,
        "MNI152NLin2009cAsym",
    )

    Path(f"{args.out_dir}/temp").mkdir(parents=True, exist_ok=True)
    # generates mni seed masks only if not found
    generate_mni_seed_masks(args)

    found_t1w_seed_masks = np.sum([
        Path(
            f"{args.out_dir}/{args.subject}_{seed[0]}_seed_T1w.nii.gz"
        ).exists() for seed in SEEDS]) == len(SEEDS)

    if not found_t1w_seed_masks:
        """
        OFFLINE work: generate T1w seed masks
        For each subject, convert each SEED mask from MNI to T1w space w ANTS
        using parcel_mni2T1w.sh script.
        Save as <args.out_dir>/<args.subject>_<seed[0]>_seed_T1w.nii.gz
        """
        print(
            "Missing T1w seed masks. Use ants to generate seed masks in"
            " subject's T1w space."
        )

    else:
        """
        Generates network masks only if all seed masks exist (MNI and T1w)
        """
        t1w_bold_list, t1w_confounds, t1w_GM_mask = get_fmri_files(args, "T1w")

        (
            seeds_ROI,
            mni_labels_mask,
            t1w_labels_mask,
        ) = make_labels_seed_masks(args, mni_GM_mask, t1w_GM_mask)

        print("Computing functional connectivity in MNI space")
        mni_fconnect_lists = compute_connectivity(
            args,
            "MNI",
            seeds_ROI,
            mni_bold_list,
            mni_GM_mask,
            mni_labels_mask,
            mni_confounds,
        )
        print("Creating network masks in MNI space")
        create_network_masks(
            args,
            "MNI",
            mni_fconnect_lists,
            mni_GM_mask,
        )

        print("Computing functional connectivity in T1w space")
        t1w_fconnect_lists = compute_connectivity(
            args,
            "T1w",
            seeds_ROI,
            t1w_bold_list,
            t1w_GM_mask,
            t1w_labels_mask,
            t1w_confounds,
        )
        print("Creating network masks in T1w space")
        create_network_masks(
            args,
            "T1w",
            t1w_fconnect_lists,
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
        default="data",
        help="Base directory for data files.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        help="Subject to use (e.g. 'sub-01').",
    )
    parser.add_argument(
        "--task_filter",
        type=str,
        default="",
        help="Regular expression to select runs.",
    )
    parser.add_argument(
        "--atlas_path",
        type=Path,
        default=Path(
            "../../../atlases/tpl-MNI152NLin2009bAsym/"
            "tpl-MNI152NLin2009bAsym_res-03_atlas-BASC_desc-197_dseg.nii.gz",
        ).resolve(),
    )

    main(parser.parse_args())
