from typing import List, Tuple, Dict, Union, Optional
from pathlib import Path

import nibabel as nib
from nibabel import Nifti1Image
from nilearn.interfaces import fmriprep
from nilearn.interfaces.bids import parse_bids_filename
from nilearn.image import (
    get_data,
    math_img,
    new_img_like,
    resample_to_img,
)
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker, NiftiMapsMasker
import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing


"""
The following functions are adapted for the CNeuroMod datasets
from the giga_connectome library, which extracts denoised time series
and computes connectomes from bids-format fMRI BOLD data
using brain parcellations in standardized space.

Source:
https://github.com/SIMEXP/giga_connectome/tree/main
"""

def get_subject_list(
    bids_dir: Path,
    subject_list: List[str] = None,
) -> List[str]:
    """
    Parse subject list if one is specified,
    otherwise return list of all subjects in bids_dir.

    Parameters
    ----------

    subject_list (optional):

        A list of BIDS compatible subject identifiers.
        E.g., ["sub-01", "sub-02", "sub-03"] or ["01", "02", "03"]
        If the prefix `sub-` is present, it will be removed.

    bids_dir :

        The fMRIPrep output directory.

    Return
    ------

    List
        BIDS subject identifier without `sub-` prefix.
    """
    # list all available subjects in dataset
    all_subjects = [
        x.name.split("-")[-1] for x in
        bids_dir.glob("sub-*/") if x.is_dir()
    ]
    if subject_list:
        checked_labels = []
        for sub_id in subject_list:
            if "sub-" in sub_id:
                sub_id = sub_id.replace("sub-", "")
            if sub_id in all_subjects:
                checked_labels.append(sub_id)
    else:
        checked_labels = all_subjects
    if not len(checked_labels) > 0:
        raise ValueError("No valid subject directory found.")

    return checked_labels


def prep_denoise_strategy(
    benchmark_strategy: Dict,
) -> dict:
    """
    Add load confound function to dictionary of parameters
    that specify the denoise strategy. These parameters are
    designed to pass to load_confounds_strategy.
    """
    lc_function = getattr(fmriprep, benchmark_strategy["function"])
    benchmark_strategy.update({"function": lc_function})

    return benchmark_strategy


def make_parcel(
    parcellation: Nifti1Image,
    epi_mask: Nifti1Image,
    gm_masking: bool,
)-> Nifti1Image:
    """
    Resample parcellation to EPI (functional) mask.
    If vowelwise timeseries are extracted from a binary mask,
    apply grey matter mask to parcellation for tighter voxel selection.
    """
    subject_parcel = resample_to_img(
        parcellation, epi_mask, interpolation="nearest",
    )
    if gm_masking:
        subject_parcel = new_img_like(
            epi_mask,
            (get_data(subject_parcel)*get_data(epi_mask)).astype("int8"),
        )

    return subject_parcel


def merge_masks(
    epi_mask: Nifti1Image,
    gm_path: Path,
    resample_gm: bool,
    use_func_mask: bool,
)-> Nifti1Image:
    """.

    Combine task-derived subject epi mask and grey matter mask into one GM mask.
    Adapted from https://github.com/SIMEXP/giga_connectome/blob/22a4ae09f647870d576ead2a73799007c1f8159d/giga_connectome/mask.py#L65
    """
    gm_mask_nii = nib.load(gm_path)

    # use wider GM mask to extract parcel signal
    if resample_gm:
        # resample template grey matter mask to subject's functional mask
        gm_mask_nii = new_img_like(
            gm_mask_nii,
            get_data(gm_mask_nii).astype(np.float64),
        )
        gm = nib.squeeze_image(
            resample_to_img(
                source_img=gm_mask_nii,
                target_img=epi_mask,
                interpolation="continuous",
            ),
        )
        # steps adapted from nilearn.images.fetch_icbm152_brain_gm_mask
        gm_mask = (get_data(gm) > 0.2).astype("int8")
        gm_mask = binary_closing(gm_mask, iterations=2)
        gm_mask_nii = new_img_like(gm, gm_mask)

    if use_func_mask:
        # combine both functional and grey matter masks into one
        return math_img(
            "img1 & img2",
            img1=epi_mask,
            img2=gm_mask_nii,
        )
    else:
        return gm_mask_nii


def denoise_nifti_voxel(
    strategy: dict,
    subject_mask: Nifti1Image, #Union[str, Path],
    standardize: str,
    smoothing_fwhm: float,
    img: str,
) -> Nifti1Image:
    """Denoise voxel level data per nifti image.
    Adapted from https://github.com/SIMEXP/giga_connectome/blob/22a4ae09f647870d576ead2a73799007c1f8159d/giga_connectome/denoise.py#L91C1-L138C1

    Parameters
    ----------
    strategy : dict
        Denoising strategy parameter to pass to load_confounds_strategy.
    subject_masker : Union[str, Path, Nifti1Image]
        Subject EPI grey matter mask or Path to it.
    standardize : str
        If 'zscore_sample', zscore the data. If 'psc', convert
        the data to percent signal change. If False, do not standardize.
    smoothing_fwhm : float
        Smoothing kernel size in mm.
    img : str
        Path to the nifti image to denoise.

    Returns
    -------
    Nifti1Image
        Denoised nifti image.
    """
    cf, sm = strategy["function"](img, **strategy["parameters"])
    if _check_exclusion(cf, sm):
        return None

    subject_masker = NiftiMasker(
        mask_img=subject_mask,
        detrend=False,
        standardize=standardize,
        smoothing_fwhm=smoothing_fwhm,
    )

    time_series_voxel = subject_masker.fit_transform(
        img, confounds=cf, sample_mask=sm
    )
    denoised_img = subject_masker.inverse_transform(time_series_voxel)
    return denoised_img


def _check_exclusion(
    reduced_confounds: pd.DataFrame, sample_mask: Optional[np.ndarray]
) -> bool:
    """For scrubbing based strategy, check if regression can be performed."""
    if sample_mask is not None:
        kept_vol = len(sample_mask)
    else:
        kept_vol = reduced_confounds.shape[0]
    # if more noise regressors than volume, this data is not denoisable
    remove = kept_vol < reduced_confounds.shape[1]
    return remove


def parse_bids_name(img: str) -> str:
    """Get subject, session, and specifier for a fMRIPrep output."""
    reference = parse_bids_filename(img)
    session = reference.get("ses", None)
    run = reference.get("run", None)
    specifier = f"task-{reference['task']}"
    if isinstance(session, str):
        session = f"ses-{session}"
        specifier = f"{session}_{specifier}"

    if isinstance(run, str):
        specifier = f"{specifier}_run-{run}"
    return session, specifier
