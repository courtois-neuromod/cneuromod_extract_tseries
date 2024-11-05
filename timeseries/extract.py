from typing import List, Tuple, Dict, Union
import glob, os, subprocess
from pathlib import Path

import h5py
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker, NiftiMasker
from nilearn.masking import compute_multi_epi_mask
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from timeseries import utils


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
        self.bids_dir = Path(
            f"{self.config.data_dir}/{self.config.dset_name}.fmriprep"
        ).resolve()
        self.subjects = utils.get_subject_list(
            self.bids_dir,
            self.config.subject_list,
        )
        self.set_paths()
        self.standardize = "zscore_sample"
        self.strategy = utils.prep_denoise_strategy(
            dict(self.config.denoise),
        )
        self.set_compression()


    def set_paths(self: "ExtractionAnalysis") -> None:
        """.

        Set analysis masks, parcellations and output paths.
        """
        # Set mask paths
        # Grey matter mask(s) (template or subject-specific)
        self.gm_path = None if self.config.gm_path is None else self._prep_paths(
            self.config.gm_path, self.config.use_template_gm, "grey matter mask")

        # Parcellation(s)/ROI mask(s). Template or subject-specific
        self.parcellation_path = self._prep_paths(
            self.config.parcellation,
            self.config.use_template_parcel,
            "parcellation",
        )

        # Set output path
        self.output_dir = Path(
            f"{self.config.output_dir}/{self.config.dset_name}/"
            f"{self.config.parcel_name}_{self.config.desc}"
        ).resolve()
        for s in self.subjects:
            Path(
                f"{self.output_dir}/sub-{s}/func"
            ).mkdir(parents=True, exist_ok=True)


    def _prep_paths(
        self: "ExtractionAnalysis",
        file_path: str,
        use_template: bool,
        file_type: str,
    ) -> Union[Path, dict]:
        """.

        Set path(s) to parcellation or GM mask (MNI template or subject-specific).
        """
        if use_template:
            prepped_path = Path(f"{file_path}").resolve()
            if not prepped_path.exists():
                raise ValueError(
                    f"{file_type} not found."
                )
        else:
            prepped_path = {
                f"sub-{s}": Path(
                    f"{file_path.replace('*', s)}"
                ).resolve() for s in self.subjects
            }
            miss_count = np.sum(
                [not v.exists() for k, v in prepped_path.items()]
            )
            if miss_count > 0:
                raise ValueError(
                    f"{miss_count} {file_type}s not found."
                )

        return prepped_path


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

            subject_parcellation, subject_epi_mask = self.make_subject_parcel(
                subject,
                mask_list,
            )
            self.extract_subject_timeseries(
                subject,
                bold_list,
                subject_parcellation,
                subject_epi_mask,
            )


    def compile_bold_list(
        self: "ExtractionAnalysis",
        subject: str,
    ) -> Tuple[list, list]:
        """.

        Compile list of subject's bold and functional brain mask files.
        """
        if self.config.space not in ["MNI152NLin2009cAsym", "T1w"]:
            raise ValueError(
                "Select 'MNI152NLin2009cAsym' or 'T1w' as template."
            )

        found_mask_list = sorted(
            glob.glob(
                f"{self.bids_dir}/sub-{subject}/"
                f"ses-*/func/*{self.config.space}"
                "*_mask.nii.gz",
                ),
            )

        bold_list = []
        mask_list = []
        for fm in found_mask_list:
            identifier = fm.split('/')[-1].split('_space')[0]
            sub, ses = identifier.split('_')[:2]

            bpath = sorted(glob.glob(
                f"{self.bids_dir}/{sub}/{ses}/func/{identifier}"
                f"*{self.config.space}*_desc-preproc_*bold.nii.gz"
            ))

            if len(bpath) == 1 and Path(bpath[0]).exists():
                bold_list.append(bpath[0])
                mask_list.append(fm)

        return bold_list, mask_list


    def make_subject_parcel(
        self: "ExtractionAnalysis",
        subject: str,
        mask_list: list,
    ) -> Tuple[nib.nifti1.Nifti1Image, nib.nifti1.Nifti1Image]:
        """.
        Generates subject EPI mask from all task run masks,
        confined by a grey matter mask if one is specified.
        Resamples parcellation atlas / ROI mask to the EPI space.
        Returns subject-specific parcellation / ROI mask and EPI mask.

        For voxelwise extractions from a binary mask
        (self.config.parcel_type == mask), the extraction is confined by
        the specified grey matter mask.
        """
        subject_epi_mask = self.make_subject_EPImask(
            subject,
            mask_list,
        )

        l_type = 'label-' if self.config.parcel_type == "mask" else "desc-"
        subject_parcel_path = Path(
            f"{self.output_dir}/sub-{subject}/func/sub-{subject}_"
            f"task-{self.config.dset_name}_space-{self.config.space}_"
            f"atlas-{self.config.parcel_name}_{l_type}{self.config.desc}_"
            f"{self.config.parcel_type}.nii.gz"
        )
        if subject_parcel_path.exists():
            print(
                "Loading existing subject parcellation."
            )
            subject_parcel = nib.load(subject_parcel_path)
        else:
            if self.config.use_template_parcel:
                parcellation = nib.load(self.parcellation_path)
            else:
                parcellation = nib.load(
                    self.parcellation_path[f"sub-{subject}"]
                )

            subject_parcel = utils.make_parcel(
                parcellation,
                subject_epi_mask,
                np.logical_and(
                    self.config.parcel_type == 'mask',
                    self.gm_path is not None,
                ),
            )
            nib.save(subject_parcel, subject_parcel_path)

        return subject_parcel, subject_epi_mask


    def make_subject_EPImask(
        self: "ExtractionAnalysis",
        subject: str,
        mask_list: list,
    ) -> nib.nifti1.Nifti1Image:
        """.

        Generate subject-specific EPI mask from all task runs.

        Combine the task-derived EPI functional mask with a grey matter
        mask if one is specified (self.gm_path != None),
        either subject-specific or from a standard template.
        """
        m_type = "bold" if self.gm_path is None else "boldGMOverlap"
        subject_mask_path = (
            f"{self.output_dir}/sub-{subject}/func/sub-{subject}_"
            f"task-{self.config.dset_name}_space-{self.config.space}_"
            f"label-{m_type}_mask.nii.gz"
        )

        if Path(subject_mask_path).exists():
            print(
                "Loading existing subject functional mask."
            )
            subject_epi_mask = nib.load(subject_mask_path)

        else:
            """
            Generate multi-session grey matter subject mask
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

            if self.gm_path is not None:
                subject_gm_path = self.gm_path if self.config.use_template_gm else self.gm_path[f"sub-{subject}"]

                # merge functional mask from subject's epi files w grey matter mask
                subject_epi_mask = utils.merge_masks(
                    subject_epi_mask,
                    subject_gm_path,
                    self.config.resample_gm,
                )

            nib.save(subject_epi_mask, subject_mask_path)

        return subject_epi_mask


    def prep_subject(
        self: "ExtractionAnalysis",
        subject: str,
        subject_parcellation: nib.nifti1.Nifti1Image,
    ) -> Tuple[Union[NiftiLabelsMasker, NiftiMapsMasker, NiftiMasker], str, List[str]]:
        """.

        Prepare subject-specific params.
        """
        if self.config.parcel_type == "mask":
            atlas_masker = NiftiMasker(
                mask_img=subject_parcellation,
                detrend=False,
                standardize=False,
                smoothing_fwhm=None,
            )
        elif self.config.parcel_type == "dseg":
            atlas_masker = NiftiLabelsMasker(
                labels_img=subject_parcellation,
                standardize=False,
            )
        elif self.config.parcel_type == "probseg":
            atlas_masker = NiftiMapsMasker(
                maps_img=subject_parcellation,
                standardize=False,
            )

        l_type = 'label-' if self.config.parcel_type == 'mask' else 'desc-'
        subj_tseries_path = (
            f"{self.output_dir}/sub-{subject}/func/"
            f"sub-{subject}_task-{self.config.dset_name}_"
            f"space-{self.config.space}_atlas-{self.config.parcel_name}_"
            f"{l_type}{self.config.desc}_timeseries.h5"
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
        subject_epi_mask: nib.nifti1.Nifti1Image,
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
                session, specifier = utils.parse_bids_name(img)
                print(specifier)

                if not f"{specifier}_timeseries" in processed_runs:
                    time_series = self.get_tseries(
                        img,
                        subject,
                        subject_epi_mask,
                        atlas_masker,
                    )

                    if time_series is not None:
                        self.save_tseries(
                            subj_tseries_path,
                            session,
                            specifier,
                            time_series,
                            self.comp_args,
                        )

            except:
                print(f"could not process file {img}" )


    def get_tseries(
        self: "ExtractionAnalysis",
        img: str,
        subject: str,
        subject_mask: nib.nifti1.Nifti1Image,
        atlas_masker: Union[NiftiLabelsMasker, NiftiMapsMasker, NiftiMasker],
    ) -> np.array:
        """.

        Extract timeseries from denoised volume.
        """
        img_temp = None
        if 'part-mag' in img:
            img_temp = f"{self.output_dir}/sub-{subject}/func/{os.path.basename(
                img).replace('_part-mag', '')}"
            conf = img.replace(
                f"_space-{self.config.space}_desc-preproc_part-mag_bold.nii.gz",
                "_desc-confounds_part-mag_timeseries.tsv",
            )
            conf_temp = f"{self.output_dir}/sub-{subject}/func/{os.path.basename(
                conf).replace('_part-mag', '')}"
            subprocess.run(
                f"cp {conf} {conf_temp}", shell = True, executable="/bin/bash",
            )

        denoised_img = utils.denoise_nifti_voxel(
            self.strategy,
            subject_mask,
            self.standardize,
            self.config.smoothing_fwhm,
            img,
            img_temp,
        )
        if 'part-mag' in img:
            subprocess.run(
                f"rm -f {conf_temp}", shell = True, executable="/bin/bash",
            )

        if not denoised_img:
            print(f"{img} : no volume after scrubbing")
            return None, None

        else:
            return atlas_masker.fit_transform(denoised_img).astype(np.float32)


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
