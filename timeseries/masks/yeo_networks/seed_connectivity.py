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
from nilearn.image import mean_img, resample_to_img
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker, NiftiSpheresMasker
from nilearn.masking import compute_multi_epi_mask


SEEDS = (
    ("visual", (-16, -74, 7), 1, 0.085),
    ("sensorimotor", (-41, -20, 62), 2, 0.06),
    ("dorsal_attention", (-34, -38, 44), 3, 0.075),
    ("ventral_attention", (-5, 15, 32), 4, 0.05),  # (-31, 11, 8)),
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
    gm_space = "" if space_name=="T1w" else f"_space-{space_name}"
    p = 0.5 if space_name=="T1w" else 0.4
    prob_GM_mask = nib.load(
        f"{args.data_dir}/sourcedata/smriprep/{args.subject}/"
        f"anat/{args.subject}{gm_space}_label-GM_probseg.nii.gz"
    )
    prob_GM_mask = resample_to_img(
        prob_GM_mask, func_mask, interpolation="linear",
    )

    binary_GM_array = (
        (prob_GM_mask.get_fdata() > p).astype(int)
    )*func_mask.get_fdata()

    return nib.nifti1.Nifti1Image(
        binary_GM_array.astype(int),
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
            f"{args.data_dir}/{args.subject}/"
            f"ses-*/func/*_space-{space_name}*_mask.nii.gz",
    ))
    found_masks = [p for p in found_masks if re.search(args.task_filter, p)]

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
    mni_func_mask: Nifti1Image,
) -> None:
    """ . """
    Path(f"{args.out_dir}/parcel_masks").mkdir(parents=True, exist_ok=True)
    Path(f"{args.out_dir}/binary_masks").mkdir(parents=True, exist_ok=True)
    Path(f"{args.out_dir}/temp").mkdir(parents=True, exist_ok=True)

    atlas_parcel = nib.load(args.atlas_path)
    for seed in SEEDS:
        mni_parcel_mask_path = Path(
            f"{args.out_dir}/parcel_masks/{args.subject}"
            f"_{seed[0]}_parcel_MNI.nii.gz"
        )
        if not mni_parcel_mask_path.exists():
            mni_seed_masker = NiftiSpheresMasker(
                [seed[1]],
                radius=1,
                detrend=False,
                standardize=False,
                verbose=0,
            )
            ROI_val = seed_masker.fit_transform(atlas_parcel)[0][0]

            parcel_mask = nib.nifti1.Nifti1Image(
                (atlas_parcel.get_fdata() == ROI_val).astype(int),
                affine=atlas_parcel.affine,
                dtype="uint8",
            )

            #rs_parcel_mask = resample_to_img(
            #    parcel_mask, mni_func_mask, interpolation="nearest",
            #)

            #masked_parcel_array = (
            #    rs_parcel_mask.get_fdata()*mni_func_mask.get_fdata()
            #).astype(int)

            assert np.sum(parcel_mask.get_fdata()) > 0.0

            #masked_parcel_mask = nib.nifti1.Nifti1Image(
            #    masked_parcel_array,
            #    affine=mni_func_mask.affine,
            #    dtype="uint8",
            #)
            nib.save(parcel_mask, mni_parcel_mask_path)


def make_labels_seed_masks(
    args: argparse.Namespace,
    mni_GM_mask: Nifti1Image,
    t1w_GM_mask: Nifti1Image,

) -> Tuple[Dict, Nifti1Image, Nifti1Image]:
    """ . """
    seeds_ROI = {s[2]:{"ROI_name": s[0]} for s in SEEDS}

    for n_ROI, seed in seeds_ROI.items():
        mni_parcel_mask_path = Path(
            f"{args.out_dir}/parcel_masks/"
            f"{args.subject}_{seed['ROI_name']}_parcel_MNI.nii.gz"
        )
        mni_parcel_mask = nib.load(mni_parcel_mask_path)
        mni_parcel_mask = resample_to_img(
            mni_parcel_mask, mni_GM_mask, interpolation="nearest",
        )
        # TODO: need squeeze?
        seeds_ROI[n_ROI]["mni_mask"] = np.squeeze(
            mni_parcel_mask.get_fdata()*mni_GM_mask.get_fdata()*n_ROI
        )

        t1w_parcel_mask_path = Path(
            f"{args.out_dir}/parcel_masks/"
            f"{args.subject}_{seed['ROI_name']}_parcel_T1w.nii.gz"
        )
        t1w_parcel_mask = nib.load(t1w_parcel_mask_path)
        t1w_parcel_mask = resample_to_img(
            t1w_parcel_mask, t1w_GM_mask, interpolation="nearest",
        )
        seeds_ROI[n_ROI]["t1w_mask"] = (
            (t1w_parcel_mask.get_fdata() > 0.5).astype(int)
        )*t1w_GM_mask.get_fdata()*n_ROI

        # TODO: make parcel mask binary (check for nearest interp rather than linear...)
        # TODO: make it binary (check for nearest interp rather than linear...)
        # TODO: mask parcel with GM mask (T1w space)
        # TODO: QC (notebook), check (visualize) that some voxels left...
        t1w_seed_max = np.max(t1w_seed_mask.get_fdata())
        seeds_ROI[n_ROI]["t1w_mask"] = (
            t1w_seed_mask.get_fdata() == t1w_seed_max
        ).astype(int)*n_ROI

    mni_labels_mask = nib.nifti1.Nifti1Image(
        np.sum(np.stack(
            [x["mni_mask"] for x in seeds_ROI.values()], axis=-1
        ), axis=-1),
        affine=mni_GM_mask.affine,
        dtype="uint8",
    )
    assert np.unique(mni_labels_mask.get_fdata()) == [x for x in seeds_ROI.keys()]

    t1w_labels_mask = nib.nifti1.Nifti1Image(
        np.sum(np.stack(
            [x["t1w_mask"] for x in seeds_ROI.values()], axis=-1
        ), axis=-1),
        affine=t1w_GM_mask.affine,
        dtype="uint8",
    )
    assert np.unique(mni_labels_mask.get_fdata()) == [x for x in seeds_ROI.keys()]

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

    connectivity_lists = {s[0]:[] for s in SEEDS}
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

        for i, r_id in enumerate(seed_masker.labels_):
            seed_name = seeds_ROI[int(r_id)]['ROI_name']
            seed_vox_corr = (
                np.dot(brain_timeseries.T, seed_timeseries[:, i])
                / seed_timeseries.shape[0]
            )

            out_path = f"{args.out_dir}/temp/{filename}_{seed_name}_{space}_connectivity.nii.gz"
            connectivity_lists[seed_name].append(out_path)
            brain_masker.inverse_transform(seed_vox_corr.T).to_filename(out_path)

    return connectivity_lists


def create_network_masks(
    args: argparse.Namespace,
    space: str,
    fconnect_lists: Dict,
    func_mask: Nifti1Image,
    GM_mask: Nifti1Image,
) -> None:
    """."""
    func_mask.to_filename(
        f"{args.out_dir}/binary_masks/{args.subject}_func-mask_{space}.nii.gz"
    )
    GM_mask.to_filename(
        f"{args.out_dir}/binary_masks/{args.subject}_GM-mask_{space}.nii.gz"
    )

    for seed in SEEDS:
        mean_connectivity = mean_img(fconnect_lists[seed[0]])
        mean_connectivity.to_filename(
            f"{args.out_dir}/binary_masks/"
            f"{args.subject}_{seed[0]}_{space}_mean-connectivity.nii.gz"

        )

        nvox = int(np.sum(func_mask.get_fdata())*seed[3])
        vox_cutoff = np.sort(mean_connectivity.get_fdata().reshape([-1]))[-nvox]
        network_arr = (mean_connectivity.get_fdata() >= vox_cutoff).astype(int)

        network_mask = nib.nifti1.Nifti1Image(
            network_arr,
            affine=mean_connectivity.affine,
            dtype="uint8",
        )
        network_mask.to_filename(
            f"{args.out_dir}/binary_masks/"
            f"{args.subject}_{seed[0]}_{space}_mask.nii.gz"
        )

        network_strict_mask = nib.nifti1.Nifti1Image(
            (network_arr*GM_mask.get_fdata()).astype(int),
            affine=mean_connectivity.affine,
            dtype="uint8",
        )
        network_strict_mask.to_filename(
            f"{args.out_dir}/binary_masks/"
            f"{args.subject}_{seed[0]}_{space}_GM-mask.nii.gz"
        )
        

def main(args: argparse.Namespace):
    """
    Derives network masks for 6 of the 7 Yeo 2011 networks in MNI and T1w space
    for each CNeuroMod subject.

    For each network, for each run, extracts time-series from the MIST-ROI
    parcel that includes a seed voxel (MNI coordinates from Yeo 2011),
    and correlates its signal with all brain voxels within a grey matter
    mask.

    Voxel-wise correlations are averaged across all task runs, and a percentage
    of voxels with the highest R scores is select to form a binary mask for
    each network.
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

    mni_bold_list, mni_confounds, mni_func_mask, mni_GM_mask = get_fmri_files(
        args,
        "MNI152NLin2009cAsym",
    )

    """
    Generates mni parcel masks (using MIST-ROI parcellation) from the network's
    seed coordinate (Yeo 2011 7-networks) only if it doesn't exist
    """
    generate_mni_parcel_masks(args, mni_func_mask, mni_GM_mask)

    found_t1w_parcel_masks = np.sum([
        Path(
            f"{args.out_dir}/parcel_masks/"
            f"{args.subject}_{seed[0]}_parcel_T1w.nii.gz"
        ).exists() for seed in SEEDS]) == len(SEEDS)

    if not found_t1w_parcel_masks:
        """
        OFFLINE work: generate T1w parcel masks
        For each subject, convert each PARCEL mask from MNI to T1w space w ANTS
        using parcel_mni2T1w.sh script.
        Save as
        <args.out_dir>/parcel_masks/<args.subject>_<seed[0]>_parcel_T1w.nii.gz
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
            t1w_bold_list, t1w_confounds, t1w_func_mask, t1w_GM_mask,
        ) = get_fmri_files(args, "T1w")

        (
            seeds_ROI,
            mni_labels_mask,
            t1w_labels_mask,
        ) = make_labels_seed_masks(
            args,
            mni_GM_mask,  # no need for func masks: GM masks are masked w func
            t1w_GM_mask,
        )

        print("Computing functional connectivity in MNI space")
        mni_fconnect_lists = compute_connectivity(
            args,
            "MNI",
            seeds_ROI,
            mni_bold_list,
            mni_func_mask,
            mni_labels_mask,
            mni_confounds,
        )
        print("Creating network masks in MNI space")
        create_network_masks(
            args,
            "MNI",
            mni_fconnect_lists,
            mni_func_mask,
            mni_GM_mask,
        )

        print("Computing functional connectivity in T1w space")
        t1w_fconnect_lists = compute_connectivity(
            args,
            "T1w",
            seeds_ROI,
            t1w_bold_list,
            t1w_func_mask, #TODO: change to GM mask? or only at export?
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
        help="Subject to use (e.g. 'sub-01').",
    )
    parser.add_argument(
        "--atlas_path",
        type=Path,
        default=Path(
            "../../../atlases/tpl-MNI152NLin2009bSym/"
            "tpl-MNI152NLin2009bSym_res-03_atlas-MIST_desc-ROI_dseg.nii.gz"
        ).resolve(),
    )
    parser.add_argument(
        "--task_filter",
        type=str,
        default="",
        help="Regular expression to select runs.",
    )


    main(parser.parse_args())
