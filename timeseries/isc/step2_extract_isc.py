#  Copyright 2017 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Intersubject correlation (ISC) analysis

Adapted from https://github.com/courtois-neuromod/cneuromod_alpha2_ISC/tree/master

Functions for computing intersubject correlation (ISC) and related
analyses (e.g., intersubject funtional correlations; ISFC), as well
as statistical tests designed specifically for ISC analyses.

The implementation is based on the work in [Hasson2004]_, [Kauppi2014]_,
[Simony2016]_, [Chen2016]_, and [Nastase2019]_.

.. [Chen2016] "Untangling the relatedness among correlations, part I:
   nonparametric approaches to inter-subject correlation analysis at the
   group level.", G. Chen, Y. W. Shin, P. A. Taylor, D. R. Glen, R. C.
   Reynolds, R. B. Israel, R. W. Cox, 2016, NeuroImage, 142, 248-259.
   https://doi.org/10.1016/j.neuroimage.2016.05.023

.. [Hasson2004] "Intersubject synchronization of cortical activity
   during natural vision.", U. Hasson, Y. Nir, I. Levy, G. Fuhrmann,
   R. Malach, 2004, Science, 303, 1634-1640.
   https://doi.org/10.1126/science.1089506

.. [Kauppi2014] "A versatile software package for inter-subject
   correlation based analyses of fMRI.", J. P. Kauppi, J. Pajula,
   J. Tohka, 2014, Frontiers in Neuroinformatics, 8, 2.
   https://doi.org/10.3389/fninf.2014.00002

.. [Simony2016] "Dynamic reconfiguration of the default mode network
   during narrative comprehension.", E. Simony, C. J. Honey, J. Chen, O.
   Lositsky, Y. Yeshurun, A. Wiesel, U. Hasson, 2016, Nature Communications,
   7, 12141. https://doi.org/10.1038/ncomms12141

.. [Nastase2019] "Measuring shared responses across subjects using
   intersubject correlation." S. A. Nastase, V. Gazzola, U. Hasson,
   C. Keysers, 2019, Social Cognitive and Affective Neuroscience, 14,
   667-685. https://doi.org/10.1093/scan/nsz037
"""

# Authors: Sam Nastase, Christopher Baldassano, Qihong Lu,
#          Mai Nguyen, and Mor Regev
# Princeton University, 2018

import argparse
import logging
from pathlib import Path

import h5py
import nibabel as nib
from nilearn.masking import unmask
import numpy as np
from scipy.spatial.distance import squareform
logger = logging.getLogger(__name__)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subjects",
        nargs='+',
        required=True,
        help="List of subjects to include. E.g., [01,02,03]",
    )
    parser.add_argument(
        "--idir",
        type=str,
        required=True,
        help="absolute path to data directory",
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


def build_dset(args):

    all_epis = {}

    dn = "Simple" if args.use_simple else ""
    for s in args.subjects:
        ipath = Path(
            f"{args.idir}/input/sub-{s}_task-friends_space_{args.space}_"
            f"season-{args.season}_desc-fwhm{args.fwhm}{dn}_bold.h5"
        )
        s_data = h5py.File(ipath, "r")
        for epi in list(s_data.keys()):
            if epi not in all_epis:
                all_epis[epi] = {
                    s: np.array(s_data[epi]).shape[0]
                }
            else:
                all_epis[epi][s] = np.array(s_data[epi]).shape[0]
        s_data.close()

    epi_list = [
        k for k, v in all_epis.items() if np.logical_and(
            sorted(v.keys()) == args.subjects,
            len(np.unique([v[x] for x in list(v.keys())]).tolist()) == 1,
        )]

    data_list = []
    for i, s in enumerate(args.subjects):
        s_data = h5py.File(
            f"{args.idir}/input/sub-{s}_task-friends_space_{args.space}_"
            f"season-{args.season}_desc-fwhm{args.fwhm}{dn}_bold.h5",
            "r")
        sub_array = [np.array(s_data[x]).astype("float32") for x in epi_list]

        data_list.append(np.concatenate(sub_array, axis=0))

    data_array = np.stack(data_list, axis=2)
    data_array[np.isnan(data_array)] = 0.0

    return data_array


def array_correlation(x, y, axis=0):
    """Column- or row-wise Pearson correlation between two arrays
    Computes sample Pearson correlation between two 1D or 2D arrays (e.g.,
    two n_TRs by n_voxels arrays). For 2D arrays, computes correlation
    between each corresponding column (axis=0) or row (axis=1) where axis
    indexes observations. If axis=0 (default), each column is considered to
    be a variable and each row is an observation; if axis=1, each row is a
    variable and each column is an observation (equivalent to transposing
    the input arrays). Input arrays must be the same shape with corresponding
    variables and observations. This is intended to be an efficient method
    for computing correlations between two corresponding arrays with many
    variables (e.g., many voxels).
    Parameters
    ----------
    x : 1D or 2D ndarray
        Array of observations for one or more variables
    y : 1D or 2D ndarray
        Array of observations for one or more variables (same shape as x)
    axis : int (0 or 1), default: 0
        Correlation between columns (axis=0) or rows (axis=1)
    Returns
    -------
    r : float or 1D ndarray
        Pearson correlation values for input variables
    """

    # Accommodate array-like inputs
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    # Check that inputs are same shape
    if x.shape != y.shape:
        raise ValueError("Input arrays must be the same shape")

    # Transpose if axis=1 requested (to avoid broadcasting
    # issues introduced by switching axis in mean and sum)
    if axis == 1:
        x, y = x.T, y.T

    # Center (de-mean) input variables
    x_demean = x - np.mean(x, axis=0)
    y_demean = y - np.mean(y, axis=0)

    # Compute summed product of centered variables
    numerator = np.sum(x_demean * y_demean, axis=0)

    # Compute sum squared error
    denominator = np.sqrt(np.sum(x_demean ** 2, axis=0) *
                          np.sum(y_demean ** 2, axis=0))

    return numerator / denominator


def _check_timeseries_input(data):
    """Checks response time series input data (e.g., for ISC analysis)
    Input data should be a n_TRs by n_voxels by n_subjects ndarray
    (e.g., brainiak.image.MaskedMultiSubjectData) or a list where each
    item is a n_TRs by n_voxels ndarray for a given subject. Multiple
    input ndarrays must be the same shape. If a 2D array is supplied,
    the last dimension is assumed to correspond to subjects. This
    function is generally intended to be used internally by other
    functions module (e.g., isc, isfc in brainiak.isc).
    Parameters
    ----------
    data : ndarray or list
        Time series data
    Returns
    -------
    data : ndarray
        Input time series data with standardized structure
    n_TRs : int
        Number of time points (TRs)
    n_voxels : int
        Number of voxels (or ROIs)
    n_subjects : int
        Number of subjects
    """

    # Convert list input to 3d and check shapes
    if type(data) == list:
        data_shape = data[0].shape
        for i, d in enumerate(data):
            if d.shape != data_shape:
                raise ValueError("All ndarrays in input list "
                                 "must be the same shape!")
            if d.ndim == 1:
                data[i] = d[:, np.newaxis]
        data = np.dstack(data)

    # Convert input ndarray to 3d and check shape
    elif isinstance(data, np.ndarray):
        if data.ndim == 2:
            data = data[:, np.newaxis, :]
        elif data.ndim == 3:
            pass
        else:
            raise ValueError("Input ndarray should have 2 "
                             "or 3 dimensions (got {0})!".format(data.ndim))

    # Infer subjects, TRs, voxels and log for user to check
    n_TRs, n_voxels, n_subjects = data.shape
    logger.info("Assuming {0} subjects with {1} time points "
                "and {2} voxel(s) or ROI(s) for ISC analysis.".format(
                    n_subjects, n_TRs, n_voxels))

    return data, n_TRs, n_voxels, n_subjects


MAX_RANDOM_SEED = 2**32 - 1


def isc(data, pairwise=False, summary_statistic=None, tolerate_nans=True):
    """Intersubject correlation

    For each voxel or ROI, compute the Pearson correlation between each
    subject's response time series and other subjects' response time series.
    If pairwise is False (default), use the leave-one-out approach, where
    correlation is computed between each subject and the average of the other
    subjects. If pairwise is True, compute correlations between all pairs of
    subjects. If summary_statistic is None, return N ISC values for N subjects
    (leave-one-out) or N(N-1)/2 ISC values for each pair of N subjects,
    corresponding to the upper triangle of the pairwise correlation matrix
    (see scipy.spatial.distance.squareform). Alternatively, use either
    'mean' or 'median' to compute summary statistic of ISCs (Fisher Z will
    be applied if using mean). Input data should be a n_TRs by n_voxels by
    n_subjects array (e.g., brainiak.image.MaskedMultiSubjectData) or a list
    where each item is a n_TRs by n_voxels ndarray for a given subject.
    Multiple input ndarrays must be the same shape. If a 2D array is supplied,
    the last dimension is assumed to correspond to subjects. If only two
    subjects are supplied, simply compute Pearson correlation (precludes
    averaging in leave-one-out approach, and does not apply summary statistic).
    When using leave-one-out approach, NaNs are ignored when computing mean
    time series of N-1 subjects (default: tolerate_nans=True). Alternatively,
    you may supply a float between 0 and 1 indicating a threshold proportion
    of N subjects with non-NaN values required when computing the average time
    series for a given voxel. For example, if tolerate_nans=.8, ISCs will be
    computed for any voxel where >= 80% of subjects have non-NaN values,
    while voxels with < 80% non-NaN values will be assigned NaNs. If set to
    False, NaNs are not tolerated and voxels with one or more NaNs among the
    N-1 subjects will be assigned NaN. Setting tolerate_nans to True or False
    will not affect the pairwise approach; however, if a threshold float is
    provided, voxels that do not reach this threshold will be excluded. Note
    that accommodating NaNs may be notably slower than setting tolerate_nans to
    False. Output is an ndarray where the first dimension is the number of
    subjects or pairs and the second dimension is the number of voxels (or
    ROIs). If only two subjects are supplied or a summary statistic is invoked,
    the output is a ndarray n_voxels long.

    The implementation is based on the work in [Hasson2004]_.

    Parameters
    ----------
    data : list or ndarray (n_TRs x n_voxels x n_subjects)
        fMRI data for which to compute ISC

    pairwise : bool, default: False
        Whether to use pairwise (True) or leave-one-out (False) approach

    summary_statistic : None or str, default: None
        Return all ISCs or collapse using 'mean' or 'median'

    tolerate_nans : bool or float, default: True
        Accommodate NaNs (when averaging in leave-one-out approach)

    Returns
    -------
    iscs : subjects or pairs by voxels ndarray
        ISC for each subject or pair (or summary statistic) per voxel

    """

    # Check response time series input format
    data, n_TRs, n_voxels, n_subjects = _check_timeseries_input(data)

    # No summary statistic if only two subjects
    if n_subjects == 2:
        logger.info("Only two subjects! Simply computing Pearson correlation.")
        summary_statistic = None

    # Check tolerate_nans input and use either mean/nanmean and exclude voxels
    if tolerate_nans:
        mean = np.nanmean
    else:
        mean = np.mean
    data, mask = _threshold_nans(data, tolerate_nans)

    # Compute correlation for only two participants
    if n_subjects == 2:

        # Compute correlation for each corresponding voxel
        iscs_stack = array_correlation(data[..., 0],
                                       data[..., 1])[np.newaxis, :]

    # Compute pairwise ISCs using voxel loop and corrcoef for speed
    elif pairwise:

        # Swap axes for np.corrcoef
        data = np.swapaxes(data, 2, 0)

        # Loop through voxels
        voxel_iscs = []
        for v in np.arange(data.shape[1]):
            voxel_data = data[:, v, :]

            # Correlation matrix for all pairs of subjects (triangle)
            iscs = squareform(np.corrcoef(voxel_data), checks=False)
            voxel_iscs.append(iscs)

        iscs_stack = np.column_stack(voxel_iscs)

    # Compute leave-one-out ISCs
    elif not pairwise:

        # Loop through left-out subjects
        iscs_stack = []
        for s in np.arange(n_subjects):

            # Correlation between left-out subject and mean of others
            iscs_stack.append(array_correlation(
                data[..., s],
                mean(np.delete(data, s, axis=2), axis=2)))

        iscs_stack = np.array(iscs_stack)

    # Get ISCs back into correct shape after masking out NaNs
    iscs = np.full((iscs_stack.shape[0], n_voxels), np.nan)
    iscs[:, np.where(mask)[0]] = iscs_stack

    # Summarize results (if requested)
    if summary_statistic:
        iscs = compute_summary_statistic(iscs,
                                         summary_statistic=summary_statistic,
                                         axis=0)[np.newaxis, :]

    # Throw away first dimension if singleton
    if iscs.shape[0] == 1:
        iscs = iscs[0]

    return iscs


def _check_isc_input(iscs, pairwise=False):
    """Checks ISC inputs for statistical tests

    Input ISCs should be n_subjects (leave-one-out approach) or
    n_pairs (pairwise approach) by n_voxels or n_ROIs array or a 1D
    array (or list) of ISC values for a single voxel or ROI. This
    function is only intended to be used internally by other
    functions in this module (e.g., bootstrap_isc, permutation_isc).

    Parameters
    ----------
    iscs : ndarray or list
        ISC values

    Returns
    -------
    iscs : ndarray
        Array of ISC values

    n_subjects : int
        Number of subjects

    n_voxels : int
        Number of voxels (or ROIs)
    """

    # Standardize structure of input data
    if type(iscs) == list:
        iscs = np.array(iscs)[:, np.newaxis]

    elif isinstance(iscs, np.ndarray):
        if iscs.ndim == 1:
            iscs = iscs[:, np.newaxis]

    # Check if incoming pairwise matrix is vectorized triangle
    if pairwise:
        try:
            test_square = squareform(iscs[:, 0], force='tomatrix')
            n_subjects = test_square.shape[0]
        except ValueError:
            raise ValueError("For pairwise input, ISCs must be the "
                             "vectorized triangle of a square matrix.")
    elif not pairwise:
        n_subjects = iscs.shape[0]

    # Infer subjects, voxels and print for user to check
    n_voxels = iscs.shape[1]
    logger.info("Assuming {0} subjects with and {1} "
                "voxel(s) or ROI(s) in bootstrap ISC test.".format(n_subjects,
                                                                   n_voxels))

    return iscs, n_subjects, n_voxels


def compute_summary_statistic(iscs, summary_statistic='mean', axis=None):
    """Computes summary statistics for ISCs

    Computes either the 'mean' or 'median' across a set of ISCs. In the
    case of the mean, ISC values are first Fisher Z transformed (arctanh),
    averaged, then inverse Fisher Z transformed (tanh).

    The implementation is based on the work in [SilverDunlap1987]_.

    .. [SilverDunlap1987] "Averaging correlation coefficients: should
       Fisher's z transformation be used?", N. C. Silver, W. P. Dunlap, 1987,
       Journal of Applied Psychology, 72, 146-148.
       https://doi.org/10.1037/0021-9010.72.1.146

    Parameters
    ----------
    iscs : list or ndarray
        ISC values

    summary_statistic : str, default: 'mean'
        Summary statistic, 'mean' or 'median'

    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.

    Returns
    -------
    statistic : float or ndarray
        Summary statistic of ISC values

    """

    if summary_statistic not in ('mean', 'median'):
        raise ValueError("Summary statistic must be 'mean' or 'median'")

    # Compute summary statistic
    if summary_statistic == 'mean':
        statistic = np.tanh(np.nanmean(np.arctanh(iscs), axis=axis))
    elif summary_statistic == 'median':
        statistic = np.nanmedian(iscs, axis=axis)

    return statistic


def _threshold_nans(data, tolerate_nans):
    """Thresholds data based on proportion of subjects with NaNs

    Takes in data and a threshold value (float between 0.0 and 1.0) determining
    the permissible proportion of subjects with non-NaN values. For example, if
    threshold=.8, any voxel where >= 80% of subjects have non-NaN values will
    be left unchanged, while any voxel with < 80% non-NaN values will be
    assigned all NaN values and included in the nan_mask output. Note that the
    output data has not been masked and will be same shape as the input data,
    but may have a different number of NaNs based on the threshold.

    Parameters
    ----------
    data : ndarray (n_TRs x n_voxels x n_subjects)
        fMRI time series data

    tolerate_nans : bool or float (0.0 <= threshold <= 1.0)
        Proportion of subjects with non-NaN values required to keep voxel

    Returns
    -------
    data : ndarray (n_TRs x n_voxels x n_subjects)
        fMRI time series data with adjusted NaNs

    nan_mask : ndarray (n_voxels,)
        Boolean mask array of voxels with too many NaNs based on threshold

    """

    nans = np.all(np.any(np.isnan(data), axis=0), axis=1)

    # Check tolerate_nans input and use either mean/nanmean and exclude voxels
    if tolerate_nans is True:
        logger.info("ISC computation will tolerate all NaNs when averaging")

    elif type(tolerate_nans) is float:
        if not 0.0 <= tolerate_nans <= 1.0:
            raise ValueError("If threshold to tolerate NaNs is a float, "
                             "it must be between 0.0 and 1.0; got {0}".format(
                                 tolerate_nans))
        nans += ~(np.sum(~np.any(np.isnan(data), axis=0), axis=1) >=
                  data.shape[-1] * tolerate_nans)
        logger.info("ISC computation will tolerate voxels with at least "
                    "{0} non-NaN values: {1} voxels do not meet "
                    "threshold".format(tolerate_nans,
                                       np.sum(nans)))

    else:
        logger.info("ISC computation will not tolerate NaNs when averaging")

    mask = ~nans
    data = data[:, mask, :]

    return data, mask


def save_isc(args, isc_results):

    tpl_mask = nib.load(
        f"{args.idir}/input/tpl-MNI152NLin2009cAsym_"
        "res-02_desc-brain_mask.nii.gz"
    )
    Path(f"{args.idir}/output").mkdir(parents=True, exist_ok=True)

    dn = "Simple" if args.use_simple else ""
    for i, snum in enumerate(args.subjects):
        nib.save(
            unmask(isc_imgs[i, :], tpl_mask),
            f"{args.idir}/output/sub-{snum}_task-friends_space_{args.space}_"
            f"season-{args.season}_stats-ISC_desc-fwhm{args.fwhm}{dn}_"
            "statseries.nii.gz",
        )


if __name__ == "__main__":

    args = get_arguments()

    data_array = build_dset(args)
    isc_results = isc(data_array, pairwise=False)

    save_isc(args, isc_results)


# import glob
# import numpy as np
# import nibabel as nib
# from brainiak.isc import isc
# from brainiak import image, io

# subjects = ['sub-01', 'sub-02', 'sub-03',
#             'sub-04', 'sub-05', 'sub-06']
# mask_name = 'tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz'
# brain_mask = io.load_boolean_mask(mask_name)
# coords = np.where(brain_mask)
# brain_nii = nib.load(mask_name)

# tasks = ['bourne', 'life']

# for task in tasks:
#     files = sorted(glob.glob(f'*{task}*fwhm*_bold.nii.gz'))

#     images = io.load_images(files)
#     masked_imgs = image.mask_images(images, brain_mask)

#     # compute LOO ISC
#     bold_imgs = image.MaskedMultiSubjectData.from_masked_images(
#         masked_imgs, 6)
#     bold_imgs[np.isnan(bold_imgs)] = 0
#     isc_imgs = isc(bold_imgs, pairwise=False)

#     # save ISC maps per subject
#     for n, subj in enumerate(subjects):

#         # Make the ISC output a volume
#         isc_vol = np.zeros(brain_nii.shape)
#         # Map the ISC data for the first participant into brain space
#         isc_vol[coords] = isc_imgs[n, :]
#         # make a nii image of the isc map
#         isc_nifti = nib.Nifti1Image(
#             isc_vol, brain_nii.affine, brain_nii.header
#         )

#         # Save the ISC data as a volume
#         isc_map_path = f'ISC_{task}_{subj}.nii.gz'
#         nib.save(isc_nifti, isc_map_path)

#     # free up memory
#     del bold_imgs, isc_imgs
