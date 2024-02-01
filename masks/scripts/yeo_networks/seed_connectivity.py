import os
import re
from glob import glob
from pathlib import Path
import argparse

import numpy as np
import nibabel as nib
import pandas as pd
import pickle as pk
from tqdm import tqdm
import nilearn.interfaces
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker, NiftiSpheresMasker

# TODO: adapt / intregrate functions below
#from src.data.load_data import load_params
#from src.tools import load_model
#from src.data.preprocess import LOAD_CONFOUNDS_PARAMS
from src.models.predict_model import predict_horizon

SEEDS = (
    ("visual", (-16, -74, 7)),
    ("sensorimotor", (-41, -20, 62)),
    ("dorsal_attention", (-34, -38, 44)),
    ("ventral_attention", (-5, 15, 32)),  # (-31, 11, 8)),
    ("fronto-parietal", (-40, 50, 7)),
    ("default-mode", (-7, -52, 26)),
)

HORIZON = 6

LOAD_CONFOUNDS_PARAMS = {
    "strategy": ["motion", "high_pass", "wm_csf", "global_signal"],
    "motion": "basic",
    "wm_csf": "basic",
    "global_signal": "basic",
    "demean": True,
}

def main(args):
    """
    Script adapted from
    https://github.com/courtois-neuromod/fmri-generator/blob/master/scripts/seed_connectivity.py

    Based on nilearn tutorial
    https://nilearn.github.io/stable/auto_examples/03_connectivity/plot_seed_to_voxel_correlation.html
    """

    file_list = glob(
        f"{args.data_dir}/{args.subject}/ses-*/func/"
        "*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )
    file_list = [p for p in file_list if re.search(args.task_filter, p)]
    confounds, _ = nilearn.interfaces.fmriprep.load_confounds(file_list, **LOAD_CONFOUNDS_PARAMS)

    #model_path = (
    #    args.model if os.path.splitext(args.model)[1] else os.path.join(args.model, "model.pkl")
    #)
    #model = load_model(model_path)
    #params = load_params(os.path.join(os.path.dirname(model_path), "params.json"))
    #batch_size = params["batch_size"] if "batch_size" in params else 100

    seeds_ROI = {}
    for seed in SEEDS:
        seed_masker = NiftiSpheresMasker(
            [seed[1]],
            radius=1,
            detrend=False,
            standardize=False,
            verbose=0,
        )
        f_id = seed_masker.fit_transform(args.atlas_path)
        assert f_id.shape == (1, 1)
        # -1 because ROIs are numbered from 1
        ROI_id = int(f_id[0][0]) - 1
        #ROI_id = int(seed_masker.fit_transform(args.atlas_path)) - 1  # deprec
        seeds_ROI[seed[0]] = ROI_id

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for i, file_path in tqdm(enumerate(file_list)):
        #predictions = {}
        filename = os.path.split(file_path)[1].replace(".nii.gz", "")

        #_, pred, atlas_series = predict_horizon(
        #    model, params["seq_length"], HORIZON, args.data_file, filename, batch_size
        #)

        brain_mask_path = file_path.replace("desc-preproc_bold", "desc-brain_mask")

        detrend = "cosine00" not in confounds[i].columns
        brain_masker = NiftiMasker(
            mask_img=brain_mask_path,
            detrend=detrend,
            standardize="zscore_sample",
            smoothing_fwhm=5,
        )

        brain_time_series = brain_masker.fit_transform(
            file_path,
            confounds=confounds[i],
        )
        #brain_time_series = brain_time_series[params["seq_length"] :]
        for seed, n_ROI in seeds_ROI.items():
            seed_time_series = atlas_series[:, n_ROI, 0]
            seed_vox_corr = (
                np.dot(brain_time_series[: -HORIZON + 1].T, seed_time_series)
                / seed_time_series.shape[0]
            )
            out_path = os.path.join(args.out_dir, f"{filename}_{seed}_original_connectivity.nii.gz")
            brain_masker.inverse_transform(seed_vox_corr.T).to_filename(out_path)
            #for lag in range(HORIZON):
            #    seed_time_series = pred[:, n_ROI, lag]
            #    last_vol = len(brain_time_series) - HORIZON + 1 + lag
            #    seed_vox_corr = (
            #        np.dot(brain_time_series[lag:last_vol].T, seed_time_series)
            #        / seed_time_series.shape[0]
            #    )
            #    out_path = os.path.join(
            #        args.out_dir, f"{filename}_{seed}_prediction_{lag}_connectivity.nii.gz"
            #    )
            #    brain_masker.inverse_transform(seed_vox_corr.T.astype(np.float32)).to_filename(
            #        out_path
            #    )


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
    # TODO: needed?
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Path to the hdf5 file with data projected on atlas.",
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
