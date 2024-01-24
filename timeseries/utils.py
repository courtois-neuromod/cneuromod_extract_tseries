from typing import List, Tuple, Dict, Union, Optional
from pathlib import Path

from nibabel import Nifti1Image
from nilearn.interfaces import fmriprep
from nilearn.interfaces.bids import parse_bids_filename
from nilearn.image import (
    get_data,
    load_img,
    math_img,
    new_img_like,
    resample_to_img,
)
from nilearn.maskers import NiftiMasker
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

PRESET_STRATEGIES = [
    "simple",
    "simple+gsr",
    "scrubbing.2",
    "scrubbing.2+gsr",
    "scrubbing.5",
    "scrubbing.5+gsr",
    "acompcor50",
    "icaaroma",
]


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


def parse_standardize_options(
    standardize: str,
) -> Union[str, bool]:
    # TODO: update standardization choices based on nilearn warnings
    if standardize not in ["zscore", "psc"]:
        raise ValueError(f"{standardize} is not a valid standardize strategy.")
    if standardize == "psc":
        return standardize
    else:
        return True


def prep_denoise_strategy(
    benchmark_strategy: Dict,
) -> dict:
    """
    Add load confound function to dictionary of parameters
    that specify the denoise strategy. These parameters are
    designed to pass to load_confounds_strategy.

    Parameters
    ---------
    benchmark_strategy : Dict
        Denoising strategy parameters specified in
        ./config/denoise/<strategy_name>.yaml config files.
        Strategy choices include:
        simple, simple+gsr, scrubbing.5, scrubbing.5+gsr, \
        scrubbing.2, scrubbing.2+gsr, acompcor50, icaaroma.

        For custom parameterization, save your own strategy config file \
        as ./config/denoise/{my_strategy}.yaml.

    Return
    ------

    dict
        Denosing strategy parameter to pass to load_confounds_strategy.
    """
    lc_function = getattr(fmriprep, benchmark_strategy["function"])
    benchmark_strategy.update({"function": lc_function})

    return benchmark_strategy


def generate_timeseries(
    masker: NiftiMasker,
    denoised_img: Nifti1Image,
) -> np.ndarray:
    """Generate denoised timeseries for one run of CNeuroMod fMRI data.

    Parameters
    ----------
    masker : NiftiMasker
        NiftiMasker instance for extracting time series.

    denoised_img : Nifti1Image
        Denoised functional image.

    Returns
    -------
    np.ndarray
        The masked time series data array.
    """
    time_series_atlas = masker.fit_transform(denoised_img)

    # convert to float 32 instead of 64
    return time_series_atlas.astype(np.float32)



def _get_consistent_masks(
    mask_imgs: List[Union[Path, str, Nifti1Image]], exclude: List[int]
) -> Tuple[List[int], List[str]]:
    """Create a list of masks that has the same affine.
    From https://github.com/SIMEXP/giga_connectome/blob/main/giga_connectome/mask.py

    Parameters
    ----------

    mask_imgs :
        The original list of functional masks

    exclude :
        List of index to exclude.

    Returns
    -------
    List of str
        Functional masks with the same affine.

    List of str
        Identifiers of scans with a different affine.
    """
    weird_mask_identifiers = []
    odd_masks = np.array(mask_imgs)[np.array(exclude)]
    odd_masks = odd_masks.tolist()
    for odd_file in odd_masks:
        identifier = Path(odd_file).name.split("_space")[0]
        weird_mask_identifiers.append(identifier)
    cleaned_func_masks = set(mask_imgs) - set(odd_masks)
    cleaned_func_masks = list(cleaned_func_masks)
    return cleaned_func_masks, weird_mask_identifiers


def _check_mask_affine(
    mask_imgs: List[Union[Path, str, Nifti1Image]]
) -> Union[list, None]:
    """Given a list of input mask images, show the most common affine matrix
    and subjects with different values.
    From https://github.com/SIMEXP/giga_connectome/blob/main/giga_connectome/mask.py

    Parameters
    ----------
    mask_imgs : :obj:`list` of Niimg-like objects
        See :ref:`extracting_data`.
        3D or 4D EPI image with same affine.

    Returns
    -------

    List or None
        Index of masks with odd affine matrix. Return None when all masks have
        the same affine matrix.
    """
    # save all header and affine info in hashable type
    header_info = {"affine": []}
    key_to_header = {}
    for this_mask in mask_imgs:
        img = load_img(this_mask)
        affine = img.affine
        affine_hashable = str(affine)
        header_info["affine"].append(affine_hashable)
        if affine_hashable not in key_to_header:
            key_to_header[affine_hashable] = affine

    if isinstance(mask_imgs[0], Nifti1Image):
        mask_imgs = np.arange(len(mask_imgs))
    else:
        mask_imgs = np.array(mask_imgs)
    # get most common values
    common_affine = max(
        set(header_info["affine"]), key=header_info["affine"].count
    )
    gc_log.info(
        f"We found {len(set(header_info['affine']))} unique affine "
        f"matrices. The most common one is "
        f"{key_to_header[common_affine]}"
    )
    odd_balls = set(header_info["affine"]) - {common_affine}
    if not odd_balls:
        return None

    exclude = []
    for ob in odd_balls:
        ob_index = [
            i for i, aff in enumerate(header_info["affine"]) if aff == ob
        ]
        gc_log.debug(
            "The following subjects has a different affine matrix "
            f"({key_to_header[ob]}) comparing to the most common value: "
            f"{mask_imgs[ob_index]}."
        )
        exclude += ob_index
    gc_log.info(
        f"{len(exclude)} out of {len(mask_imgs)} has "
        "different affine matrix. Ignore when creating group mask."
    )
    return sorted(exclude)


def merge_masks(
    subject_epi_mask: Nifti1Image,
    mni_gm_path: str,
    n_iter: int,
)-> Nifti1Image:
    """.

    Combine task-derived subject epi mask and MNI template mask into one GM mask.
    Adapted from https://github.com/SIMEXP/giga_connectome/blob/22a4ae09f647870d576ead2a73799007c1f8159d/giga_connectome/mask.py#L65
    """
    # resample MNI grey matter template mask to subject's grey matter mask
    mni_gm = nib.squeeze_image(
        resample_to_img(
            source_img=mni_gm_path,
            target_img=subject_epi_mask,
            interpolation="continuous",
        ),
    )

    # steps adapted from nilearn.images.fetch_icbm152_brain_gm_mask
    mni_gm_mask = (get_data(mni_gm) > 0.2).astype("int8")
    mni_gm_mask = binary_closing(mni_gm_mask, iterations=n_iter)
    mni_gm_mask_nii = new_img_like(mni_gm, mni_gm_mask)

    # combine both subject and template masks into one
    subject_mask_nii = math_img(
        "img1 & img2",
        img1=subject_epi_mask,
        img2=mni_gm_mask_nii,
    )

    return subject_mask_nii


def denoise_nifti_voxel(
    strategy: dict,
    subject_mask: Nifti1Image, #Union[str, Path],
    standardize: Union[str, bool],
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
    standardize : Union[str, bool]
        TODO: update based on nilearn normalizing warning
        Standardize the data. If 'zscore', zscore the data. If 'psc', convert
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

    # if high pass filter is not applied through cosines regressors,
    # then detrend
    detrend = "cosine00" not in cf.columns
    subject_masker = NiftiMasker(
        mask_img=subject_mask,
        detrend=detrend,
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
