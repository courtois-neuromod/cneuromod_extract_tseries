from typing import List, Tuple, Dict, Union
import glob
from pathlib import Path

import h5py
import nibabel as nib
from nilearn.image import (
    get_data,
    math_img,
    new_img_like,
    resample_to_img,
)
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker
from nilearn.masking import compute_multi_epi_mask
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from utils import (
    denoise_nifti_voxel,
    generate_timeseries,
    prep_denoise_strategy,
    get_subject_list,
    merge_masks,
    parse_bids_name,
    parse_standardize_options,
    _check_mask_affine,
    _get_consistent_masks,
)


class ExtractionAnalysis:
    """ExtractionAnalysis.

    This class performs the extraction of timeseries from a
    CNeuromod dataset within a custom set of parcels using
    a specified denoising strategy.
    """

    def __init__(
        self: "ExtractionAnalysis",
        config: DictConfig,
    ) -> None:
        """.

        Extracts paths and timeseries extraction parameters from the
        Hydra configuration file(s).
        """
        self.config: DictConfig = config
        self.set_params()


    def set_params(self: "ExtractionAnalysis") -> None:
        """.

        Define analysis parameters.
        """
        self.set_paths()
        self.subjects = get_subject_list(
            self.bids_dir,
            self.config.subject_list,
        )
        # define analysis params
        # TODO: update standardization, see nilearn warning
        self.standardize = parse_standardize_options(
            self.config.standardize,
        )
        self.strategy = prep_denoise_strategy(
            dict(self.config.denoise),
        )
        self.set_compression()


    def set_paths(self: "ExtractionAnalysis") -> None:
        """.

        Set all analysis input and output paths.
        """
        # Input paths
        self.bids_dir = Path(
            f"{self.config.data_dir}/{self.config.dset_name}.fmriprep"
        ).resolve()
        # grey-matter mask in MNI-space, probability segmentation
        if self.config.template == "MNI152NLin2009cAsym":
            self.template_gm_path = Path(
                f"{self.config.template_gm_path}",
            ).resolve()
            if not self.template_gm_path.exists():
                raise ValueError(
                    "Template grey matter mask not found."
                )
        # define path to parcellation atlas
        # TODO: update depending on template vs native parcellation
        if self.config.template_parcellation is not None:
            self.parcellation_path = Path(
                f"{self.config.template_parcellation}"
            ).resolve()
            if not self.parcellation_path.exists():
                raise ValueError(
                    "Template parcellation not found."
                )

        # Set output paths
        self.timeseries_dir = Path(
            f"{self.config.output_dir}/{self.config.dset_name}/timeseries"
        ).resolve()
        self.timeseries_dir.mkdir(parents=True, exist_ok=True)
        # save subject-specific grey matter mask and parcel atlas
        self.mask_dir = Path(
            f"{self.config.output_dir}/{self.config.dset_name}/subject_masks"
        ).resolve()
        self.mask_dir.mkdir(exist_ok=True, parents=True)


    def set_compression(self: "ExtractionAnalysis") -> None:
        """.

        Set compression parameters for timeseries exported in .h5 file
        """
        if self.config.compression is None:
            self.comp_args = {}
        elif self.config.compression not in ["gzip", "lzf"]:
            raise ValueError(
                "Select 'gzip', 'lzf' or 'None' as valid compression."
            )
        else:
            self.comp_args = {
                "compression": self.config.compression,
            }
            if self.config.compression == "gzip":
                if self.config.compression_opts not in range(1, 10):
                    raise ValueError(
                        "Select compression_opts in [0, 10] for gzip compression."
                    )
                self.comp_args["compression_opts"] = self.config.compression_opts


    def extract(self: "ExtractionAnalysis") -> None:
        """.

        Extract denoised timeseries from individual subject datasets.
        """
        for subject in self.subjects:
            bold_list, mask_list = self.compile_bold_list(subject)

            subject_parcellation, subject_mask = self.make_subject_parcel(
                subject,
                mask_list,
            )
            self.extract_subject_timeseries(
                subject,
                bold_list,
                subject_parcellation,
                subject_mask,
            )


    def compile_bold_list(
        self: "ExtractionAnalysis",
        subject: str,
    ) -> Tuple[list, list]:
        """.

        Compile list of subject's bold and functional brain mask files.
        """
        if self.config.template not in ["MNI152NLin2009cAsym", "T1w"]:
            raise ValueError(
                "Select 'MNI152NLin2009cAsym' or 'T1w' as template."
            )

        found_mask_list = sorted(
            glob.glob(
                f"{self.bids_dir}/sub-{subject}/"
                f"ses-*/func/*{self.config.template}"
                "*_mask.nii.gz",
                ),
            )
        if exclude := _check_mask_affine(found_mask_list, verbose=2):
            found_mask_list, __annotations__ = _get_consistent_masks(
                found_mask_list,
                exclude,
                )
            print(f"Remaining: {len(found_mask_list)} masks")

        bold_list = []
        mask_list = []
        for fm in found_mask_list:
            identifier = fm.split('/')[-1].split('_space')[0]
            sub, ses = identifier.split('_')[:2]
            bpath = sorted(glob.glob(
                f"{self.bids_dir}/{sub}/{ses}/func/"
                f"{identifier}*_desc-preproc_*bold.nii.gz"
            ))

            if len(bpath) == 1 and Path(bpath[0]).exists():
                bold_list.append(bpath[0])
                mask_list.append(fm)

        return bold_list, mask_list


    def make_subject_parcel(
        self: "ExtractionAnalysis",
        subject: str,
        mask_list: list,
    ) -> (nib.nifti1.Nifti1Image, nib.nifti1.Nifti1Image):
        """.

        Return subject-specific parcellation.
        """
        subject_mask = self.make_subjectGM_mask(
            subject,
            mask_list,
        )
        subject_parcel_path = Path(
            f"{self.mask_dir}/sub-{subject}_{self.config.template}_"
            f"{self.config.parcel_name}.nii.gz"
        )

        if subject_parcel_path.exists():
            subject_parcel = nib.load(subject_parcel_path)
        else:
            """
            Generate subject grey matter mask from task runs EPI masks,
            then resample parcellation atlas to the EPI space.
            """
            template_parcellation = nib.load(self.parcellation_path)
            subject_parcel = resample_to_img(
                template_parcellation, subject_mask, interpolation="nearest"
            )
            nib.save(subject_parcel, subject_parcel_path)

        return subject_parcel, subject_mask


    def make_subjectGM_mask(
        self: "ExtractionAnalysis",
        subject: str,
        mask_list: list,
    ) -> nib.nifti1.Nifti1Image:
        """.

        Generate subject-specific EPI grey matter mask from all task runs.

        If template is MNI152NLin2009cAsym (analysis in normalized space),
        overlay task-derived EPI grey matter mask with a MNI grey
        matter template (MNI152NLin2009cAsym template to match the template).
        """
        subject_mask_path = (
            f"{self.mask_dir}/sub-{subject}_{self.conf.dset_name}"
            f"_{self.config.template}_res-dataset_label-GM_desc_mask.nii.gz"
        )

        if Path(subject_mask_path).exists():
            print(
                "Loading existing subject grey matter mask."
            )
            return nib.load(subject_mask_path)

        else:
            """
            Generate multi-session grey matter subject mask in MNI152NLin2009cAsym
            Parameters from
            https://github.com/SIMEXP/giga_connectome/blob/main/giga_connectome/mask.py
            """
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
            print(
                f"Group EPI mask affine:\n{subject_epi_mask.affine}"
                f"\nshape: {subject_epi_mask.shape}"
            )

            if self.config.template == "MNI152NLin2009cAsym":
                # merge mask from subject's epi files w MNI template grey matter mask
                subject_epi_mask = merge_masks(
                    subject_epi_mask,
                    self.template_gm_path,
                    self.config.n_iter,
                )

            nib.save(subject_epi_mask, subject_mask_path)

            return subject_epi_mask


    def prep_subject(
        self: "ExtractionAnalysis",
        subject: str,
        subject_parcellation: nib.nifti1.Nifti1Image,
    ) -> (NiftiLabelsMasker, str, list):
        """.

        Prepare subject-specific params.
        """
        if self.config.parcel_type == "dseg":
            atlas_masker = NiftiLabelsMasker(
                labels_img=subject_parcellation,
                standardize=False,
            )
        elif self.config.parcel_type == "probseg":
            atlas_masker = NiftiMapsMasker(
                maps_img=subject_parcellation,
                standardize=False,
            )

        subj_tseries_path = (
            f"{self.timeseries_dir}/"
            f"sub-{subject}_{self.conf.dset_name}_{self.conf.template}"
            f"_BOLDtimeseries_{self.conf.parcel_name}"
            f"_{self.strategy['name']}.h5"
        )

        processed_run_list = []
        if Path(subj_tseries_path).exists():
            with h5py.File(subj_tseries_path, 'r') as f:
                sessions = [g for g in f.keys()]
                for g in sessions:
                    processed_run_list += [r for r in f[g].keys()]

        return atlas_masker, subj_tseries_path, processed_run_list


    def extract_subject_timeseries(
        self: "ExtractionAnalysis",
        subject: str,
        bold_list: list,
        subject_parcellation: nib.nifti1.Nifti1Image,
        subject_mask: nib.nifti1.Nifti1Image,
    ) -> None:
        """.

        Generate subject's run-level time series.
        """
        atlas_masker, subj_tseries_path, processed_runs = self.prep_subject(
            subject,
            subject_parcellation,
        )

        for img in tqdm(bold_list):
            try:
                session, specifier = parse_bids_name(img)
                print(specifier)

                if not f"{specifier}_timeseries" in processed_runs:
                    time_series = self.get_tseries(
                        img,
                        subject_mask,
                        atlas_masker,
                    )

                    if time_series is not None:
                        self.save_tseries(
                            subj_tseries_path,
                            session,
                            specifier,
                            time_series,
                        )

            except:
                print(f"could not process file {img}" )

        return


    def get_tseries(
        self: "ExtractionAnalysis",
        img: str,
        subject_mask: nib.nifti1.Nifti1Image,
        atlas_masker: Union[NiftiLabelsMasker, NiftiMapsMasker],
    ) -> np.array:
        """.

        Extract timeseries from denoised volume.
        """
        denoised_img = denoise_nifti_voxel(
            self.strategy,
            subject_mask,
            self.standardize,
            self.conf.smoothing_fwhm,
            img,
        )

        if not denoised_img:
            print(f"{img} : no volume after scrubbing")
            return None, None

        else:
            time_series = generate_timeseries(
                atlas_masker,
                denoised_img,
            )

            return time_series


    def save_tseries(
        self: "ExtractionAnalysis",
        subj_tseries_path: str,
        session: str,
        specifier: str,
        time_series: np.array,
        comp_args: Dict,
    ) -> None:
        """.

        Save episode's time series in .h5 file.
        """
        flag =  "a" if Path(subj_tseries_path).exists() else "w"
        with h5py.File(subj_tseries_path, flag) as f:

            group = f.create_group(session) if not session in f else f[session]

            timeseries_dset = group.create_dataset(
                f"{specifier}_timeseries",
                data=time_series,
                **comp_args,
            )
