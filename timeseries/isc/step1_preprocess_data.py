"""
Adapted from https://github.com/courtois-neuromod/cneuromod_alpha2_ISC/tree/master/src/data
"""
import re, glob
import argparse
from pathlib import Path

import h5py
import numpy as np
import nibabel as nib
import nilearn.interfaces
from numpy.lib import recfunctions
from nilearn import image
from nilearn.maskers import NiftiMasker
from tqdm import tqdm


LOAD_CONFOUNDS_PARAMS = {
    "strategy": ["motion", "high_pass", "wm_csf", "global_signal"],
    "motion": "basic",
    "wm_csf": "basic",
    "global_signal": "basic",
    "demean": True,
}


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="Two-digit subject number. E.g., 01",
    )
    parser.add_argument(
        "--idir",
        type=str,
        required=True,
        help="absolute path to bold dataset",
    )
    parser.add_argument(
        "--mdir",
        type=str,
        required=True,
        help="absolute path to atlas directory",
    )
    parser.add_argument(
        "--odir",
        type=str,
        required=True,
        help="absolute path to output directory",
    )
    parser.add_argument(
        "--space",
        type=str,
        choices=["MNI152NLin2009cAsym", "T1w"],
        help="EPI space. Choose either MNI152NLin2009cAsym or T1w",
    )
    parser.add_argument(
        "--fwhm",
        type=int,
        default=6,
        help="smoothing kernel full-width at half-maximum in mm",
    )
    parser.add_argument(
        "--season",
        type=str,
        default="06",
        help="two-digit season number, eg. 06",
    )
    parser.add_argument(
        "--use_simple",
        default=False,
        action="store_true",
        help="If True, use load_confound simple denoising strategy",
    )
    return parser.parse_args()


def get_lists(args):
    snum = args.subject
    tpl_mask = Path(
        f"{args.mdir}/tpl-MNI152NLin2009cAsym/"
        "tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"
    )

    bold_list = sorted(glob.glob(
        f"{args.idir}/sub-{snum}/ses-0*/func/sub-{snum}_ses-*_task-s{args.season}*_"
        f"space-{args.space}_desc-preproc_bold.nii.gz"
    ))

    if args.use_simple:
        confound_list, _ = nilearn.interfaces.fmriprep.load_confounds(
            bold_list,
            **LOAD_CONFOUNDS_PARAMS,
        )
    else:
        confound_list = [
            recfunctions.structured_to_unstructured(_subset_confounds(
                x.replace(
                    f"_space-{args.space}_desc-preproc_bold.nii.gz",
                    "_desc-confounds_timeseries.tsv",
                )
            )) for x in bold_list
        ]

    dn = "Simple" if args.use_simple else ""
    Path(f"{args.odir}/input").mkdir(parents=True, exist_ok=True)
    out_file = Path(
        f"{args.odir}/input/sub-{args.subject}_task-friends_space_{args.space}_"
        f"season-{args.season}_desc-fwhm{args.fwhm}{dn}_bold.h5"
    )

    return bold_list, confound_list, tpl_mask, out_file


def denoise_bold(args, scan, mask, confounds, fwhm):
    """."""
    if args.use_simple:
        masker = NiftiMasker(
            mask_img=mask,
            t_r=1.49,
            standardize=True,
            detrend=False,
            smoothing_fwhm=fwhm,
        )
    else:
        masker = NiftiMasker(
            mask_img=mask,
            t_r=1.49,
            standardize=True,
            detrend=True,
            high_pass=0.01,
            low_pass=0.1,
            smoothing_fwhm=fwhm,
        )
        compcor = image.high_variance_confounds(
            scan, mask_img=mask, n_confounds=10, percentile=5.0)
        confounds = np.hstack((confounds, compcor))

    cleaned = masker.fit_transform(scan, confounds=confounds)
    #return masker.inverse_transform(cleaned)
    return cleaned.astype("float32")


def _subset_confounds(tsv):
    """
    Only retain those confounds listed in `keep_confounds`

    Parameters
    ----------
    tsv: str
        Local file path to the fMRIPrep generated confound files
    """
    keep_confounds = ['trans_x', 'trans_y', 'trans_z',
                      'rot_x', 'rot_y', 'rot_z',
                      'csf', 'white_matter']

    # load in tsv and subset to only include our desired regressors
    tsv = str(tsv)
    confounds = np.recfromcsv(tsv, delimiter='\t')
    selected_confounds = confounds[keep_confounds]
    return selected_confounds


def save_clean(args, episode, clean_bold, out_file):
    """."""
    flag =  "a" if out_file.exists() else "w"
    with h5py.File(out_file, flag) as f:
        dset = f.create_dataset(
            episode,
            data=clean_bold,
        )


if __name__ == "__main__":

    args = get_arguments()

    bold_list, confound_list, tpl_mask, out_file = get_lists(args)

    for i, bold in tqdm(enumerate(bold_list), desc="denoising bold"):
        episode = bold.split("/")[-1].split("_")[2].split("-")[-1]
        # masked dim = (TR, voxels)
        clean_bold = denoise_bold(
            args, bold, tpl_mask, confound_list[i], fwhm=args.fwhm,
        )
        save_clean(args, episode, clean_bold, out_file)

    nib.save(
        nib.load(tpl_mask),
        f"{args.odir}/input/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"
    )
