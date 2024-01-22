from typing import List, Tuple, Dict, Union
from pathlib import Path
from nilearn.interfaces.bids import parse_bids_filename
from bids.layout import Query
from bids import BIDSLayout

from omegaconf import DictConfig
from nilearn.interfaces import fmriprep


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


def get_denoise_strategy(
    strategy: str,
    strategy_dir: str = None,
) -> dict:
    """
    Select denoise strategies and associated parameters.
    The strategy parameters are designed to pass to load_confounds_strategy.

    Parameters
    ---------
    strategy : str
        Name of the provided denoising strategy options: \
        simple, simple+gsr, scrubbing.5, scrubbing.5+gsr, \
        scrubbing.2, scrubbing.2+gsr, acompcor50, icaaroma.

        For custom parameterization, save your own configuration json file \
        as {strategy}.json under {strategy_dir}.
        https://giga-connectome.readthedocs.io/en/stable/usage.html#denoising-strategy

    strategy_dir (optional): str
        Path to directory with denoise strategy .json file  \
        Default strategies are saved under ../denoise

    Return
    ------

    dict
        Denosing strategy parameter to pass to load_confounds_strategy.
    """
    if strategy in PRESET_STRATEGIES:
        config_path = Path(
            f"../denoise/{strategy}.json"
        ).resolve()
    else:
        config_path = Path(
            f"{strategy_dir}/{strategy}.json"
        ).resolve()

    if not config_path.exists():
        raise ValueError(f"No {strategy} strategy config file found.")

    with open(config_path, "r") as file:
        benchmark_strategy = json.load(file)

    lc_function = getattr(fmriprep, benchmark_strategy["function"])
    benchmark_strategy.update({"function": lc_function})

    return benchmark_strategy


def generate_timeseries(
    masker: NiftiMasker,
    denoised_img: Nifti1Image,
) -> np.ndarray:
    """Generate denoised timeseries from CNeuroMod subject's fMRI data.

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
