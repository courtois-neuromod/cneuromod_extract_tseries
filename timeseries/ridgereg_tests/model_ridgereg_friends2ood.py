import argparse
import glob
import json
import sys
from pathlib import Path

import h5py
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
import numpy as np
from scipy import spatial
from scipy.spatial.distance import cosine
import scipy.stats as stats
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold


def get_arguments() -> argparse.Namespace:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Conducts ridge regression on parcelated Friends dataset and test "
            "generalization on OOD dataset."
        ),
    )
    parser.add_argument(
        "--idir",
        type=str,
        required=True,
        help="Path to directory with input features.",
    )
    parser.add_argument(
        "--bdir",
        type=str,
        required=True,
        help="Path to directory with BOLD timeseries data.",
    )
    parser.add_argument(
        "--odir",
        type=str,
        required=True,
        help="Path to directory where output files will be saved.",
    )
    parser.add_argument(
        "--atlas",
        type=str,
        required=True,
        help="Name of the parcellation. e.g., MIST, Schaefer18",
    )
    parser.add_argument(
        "--parcel",
        type=str,
        required=True,
        help="Parcellation grain. e.g., 444, 1000Parcels7Networks",
    )
    parser.add_argument(
        "--participant",
        type=str,
        required=True,
        help="CNeuroMod participant label. E.g., sub-01.",
    )
    parser.add_argument(
        "--modalities",
        choices=['visual', 'audio', 'text'],
        help="Input modalities to include in ridge regression.",
        nargs="+",
    )
    parser.add_argument(
        "--text_features",
        choices=['text_pooled', 'text_token', 'text_OAI', 'text_1hot', 'text_1hothtk'],
        help="Text features to include in ridge regression.",
        nargs="+",
    )
    parser.add_argument(
        "--back",
        type=int,
        default=5,
        choices=range(0, 10),
        help="How far back in time (in TRs) does the input window start "
        "in relation to the TR it predicts. E.g., back = 5 means that input "
        "features are sampled starting 5 TRs before the target BOLD TR onset",
    )
    parser.add_argument(
        "--input_duration",
        type=int,
        default=3,
        choices=range(1, 5),
        help="Duration of input time window (in TRs) to predict a BOLD TR. "
        "E.g., input_duration = 3 means that input is sampled over 3 TRs "
        "to predict a target BOLD TR.",
    )
    parser.add_argument(
        "--n_split",
        type=int,
        default=8,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=None,
        help="Set seed to assign runs to train & validation sets.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Set to 1 for extra information. Default is 0.",
    )

    return parser.parse_args()


def split_episodes(
    args: argparse.Namespace,
) -> tuple:
    """.

    Assigns subject's runs to train & validation sets
    """
    friends_h5 = h5py.File(
        f"{args.bdir}/friends/{args.atlas}_{args.parcel}/{args.participant}"
        f"/func/{args.participant}_task-friends_space-MNI152NLin2009cAsym_"
        f"atlas-{args.atlas}_desc-{args.parcel}_timeseries.h5",
        "r",
    )
    # Season 3 held out for test set

    # Remaining runs assigned to train and validation sets
    r = np.random.RandomState(args.random_state)  # select season for validation set

    if args.participant == 'sub-04':
        val_season = r.choice(["s01", "s02", "s04"], 1)[0]
    else:
        val_season = r.choice(["s01", "s02", "s04", "s05", "s06"], 1)[0]
    val_set = []
    for ses in friends_h5:
        val_set += [
            x for x in friends_h5[ses] if x.split('-')[-1][:3] == val_season]
    train_set = []
    for ses in friends_h5:
        train_set += [
            x for x in friends_h5[ses] if x.split('-')[-1][:3] not in ['s03', 's07', val_season]]
    train_set = sorted(train_set)

    friends_h5.close()

    # Assign consecutive train set episodes to cross-validation groups
    lts = len(train_set)
    train_groups = np.floor(np.arange(lts)/(lts/args.n_split)).astype(int).tolist()


    # Make test sets from OOD dataset runs
    ood_h5 = h5py.File(
        f"{args.bdir}/ood/{args.atlas}_{args.parcel}/{args.participant}"
        f"/func/{args.participant}_task-ood_space-MNI152NLin2009cAsym_"
        f"atlas-{args.atlas}_desc-{args.parcel}_timeseries.h5",
        "r",
    )

    # build list of runs per OOD movie
    test_sets = {}
    mv_list = [
        "chaplin", "mononoke", "passepartout", "planetearth", "pulpfiction", "wot",
    ]
    for mv_name in mv_list:
        test_sets[mv_name] = []
        for ses in ood_h5:
            test_sets[mv_name] += [x for x in ood_h5[ses] if mv_name in x]

    ood_h5.close()

    return train_groups, train_set, val_set, test_sets


def build_audio_visual(
    idir: str,
    dset: str,
    modalities: list,
    runs: list,
    run_lengths: list,
    duration: int,
) -> np.array:
    """.

    Concatenates visual and audio features into array.
    """

    x_list = []

    for run, rl in zip(runs, run_lengths):
        run_name = run.split('-')[-1].split('_')[0]
        season: str = run_name[2]

        if dset == "friends":
            h5_path = Path(
                f"{idir}/friends_s{season}_"
                "features_visual_audio_gzip_level-4.h5"
            )
        else:
            h5_path = Path(
                f"{idir}/ood_features_visual_audio.h5"
            )

        run_input = {}
        with h5py.File(h5_path, "r") as f:
            for modality in modalities:
                run_input[modality] = np.array(f[run_name][modality])

        run_list = []

        for modality in modalities:
            for k in range(duration):
                run_list.append(
                    np.nan_to_num(
                        # default computes zscore over axis 0
                        stats.zscore(
                            run_input[modality][k:(rl+k), :]
                        )
                    )
                )

        x_list.append(np.concatenate(run_list, axis=1))

    return np.concatenate(x_list, axis=0)


def build_text(
    idir: str,
    dset: str,
    feature_list: list,
    runs: list,
    run_lengths: list,
    duration: int,
) -> np.array:

    dur = duration - 1
    #feature_list = ['text_pooled', 'text_token', 'text_OAI', 'text_1hot']
    x_dict = {k:[] for k in feature_list}

    for run, rl in zip(runs, run_lengths):
        run_name = run.split('-')[-1].split('_')[0]
        season: str = run_name[2]

        if dset == "friends":
            h5_path = Path(
                f"{idir}/friends_s{season}_"
                "features_text_gzip_level-4.h5"
            )
        else:
            h5_path = Path(
                f"{idir}/ood_features_text.h5"
            )

        with h5py.File(h5_path, "r") as f:
            for feat_type in feature_list:
                run_data = np.array(f[run_name][feat_type])[dur: dur+rl]

                # pad features array in case fewer text TRs than for BOLD data
                rdims = run_data.shape
                if len(rdims) == 1:
                    run_data = np.expand_dims(run_data, axis=1)
                    rdims = run_data.shape

                rsize = rl*rdims[1] if len(rdims) == 2 else rl*rdims[1]*rdims[2]
                run_array = np.repeat(np.nan, rsize).reshape((rl,) + rdims[1:])
                run_array[:rdims[0]] = run_data

                x_dict[feat_type].append(run_array)

    x_list = []
    for feat_type in feature_list:
        feat_data = np.concatenate(x_dict[feat_type], axis=0)
        dims = feat_data.shape

        x_list.append(
            np.nan_to_num(
                stats.zscore(
                    feat_data.reshape((-1, dims[-1])),
                    nan_policy="omit",
                    axis=0,
                )
            ).reshape(dims).reshape(dims[0], -1).astype('float32')
        )

    return np.concatenate(x_list, axis=1)


def build_X(
    args: argparse.Namespace,
    dset: str,
    runs: list,
    run_lengths: list,
) -> np.array:
    """.

    Concatenates input features across modalities into predictor array.
    """

    x_list = []

    av_modalities = [x for x in args.modalities if x in ["visual", "audio"]]
    if len(av_modalities) > 0:
        x_list.append(
            build_audio_visual(
                args.idir,
                dset,
                av_modalities,
                runs,
                run_lengths,
                args.input_duration,
            ),
        )

    if "text" in args.modalities:
        x_list.append(
            build_text(
                args.idir,
                dset,
                args.text_features,
                runs,
                run_lengths,
                args.input_duration,
            ),
        )


    if len(x_list) > 1:
        return np.concatenate(x_list, axis=1)
    else:
        return x_list[0]


def build_y(
    args: argparse.Namespace,
    dset: str,
    runs: list,
    run_groups: list = None,
) -> tuple:
    """.

    Concatenates BOLD timeseries into target array.
    """
    y_list = []
    length_list = []
    y_groups = []
    sub_h5 = h5py.File(
        f"{args.bdir}/{dset}/{args.atlas}_{args.parcel}/{args.participant}"
        f"/func/{args.participant}_task-{dset}_space-MNI152NLin2009cAsym_"
        f"atlas-{args.atlas}_desc-{args.parcel}_timeseries.h5",
        "r",
    )

    for i, run in enumerate(runs):
        ses = run.split("_")[0]
        run_ts = np.array(sub_h5[ses][run])[args.back:, :]
        length_list.append(run_ts.shape[0])
        y_list.append(run_ts)

        if run_groups is not None:
            y_groups.append(np.repeat(run_groups[i], run_ts.shape[0]))

    sub_h5.close()
    y_list = np.concatenate(y_list, axis=0)
    y_groups = np.concatenate(y_groups, axis=0) if run_groups is not None else np.array([])

    return y_list, length_list, y_groups


def train_ridgeReg(
    X: np.array,
    y: np.array,
    groups: list,
    n_splits: int,
) -> RidgeCV:
    """.

    Performs ridge regression with built-in cross-validation.
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html
    """
    alphas = np.logspace(0.1, 3, 10)
    group_kfold = GroupKFold(n_splits=n_splits)
    cv = group_kfold.split(X, y, groups)

    model = RidgeCV(
        alphas=alphas,
        fit_intercept=True,
        #normalize=False,
        cv=cv,
    )

    return model.fit(X, y)


def pairwise_acc(
    target: np.array,
    predicted: np.array,
    use_distance: bool = False,
) -> float:
    """.

    Computes Pairwise accuracy
    Adapted from: https://github.com/jashna14/DL4Brain/blob/master/src/evaluate.py
    """
    true_count = 0
    total = 0

    for i in range(0,len(target)):
        for j in range(i+1, len(target)):
            total += 1

            t1 = target[i]
            t2 = target[j]
            p1 = predicted[i]
            p2 = predicted[j]

            if use_distance:
                if cosine(t1,p1) + cosine(t2,p2) < cosine(t1,p2) + cosine(t2,p1):
                    true_count += 1

            else:
                if pearsonr(t1,p1)[0] + pearsonr(t2,p2)[0] > pearsonr(t1,p2)[0] + pearsonr(t2,p1)[0]:
                    true_count += 1

    return (true/total)


def pearson_corr(
    target: np.array,
    predicted: np.array,
) -> np.array:
    """.

    Calculates pearson R between predictions and targets.
    """
    r_vals = []
    for i in range(len(target)):
        r_val, _  = pearsonr(target[i], predicted[i])
        r_vals.append(r_val)

    return np.array(r_vals)


def export_images(
    args: argparse.Namespace,
    modal_names: str,
    results: dict,
) -> None:
    """.

    Exports RR parcelwise scores as nifti files with
    subject-specific atlas used to extract timeseries.
    """
    friends_atlas_path = Path(
        f"{args.bdir}/friends/{args.atlas}_{args.parcel}/{args.participant}"
        f"/func/{args.participant}_task-friends_space-MNI152NLin2009cAsym_"
        f"atlas-{args.atlas}_desc-{args.parcel}_dseg.nii.gz"
    )
    friends_atlas_masker = NiftiLabelsMasker(
        labels_img=friends_atlas_path,
        standardize=False,
    )
    friends_atlas_masker.fit()

    # map Pearson correlations onto brain parcels
    for s in ['train', 'val']:
        nii_file = friends_atlas_masker.inverse_transform(
            np.array(results["parcelwise"][f"{s}_R2"]),
        )
        nib.save(
            nii_file,
            f"{args.odir}/{args.participant}_dset-friends2OOD_{args.atlas}_"
            f"{args.parcel}_RidgeReg{modal_names}_R2_{s}_friends.nii.gz",
        )

    ood_atlas_path = Path(
        f"{args.bdir}/ood/{args.atlas}_{args.parcel}/{args.participant}"
        f"/func/{args.participant}_task-ood_space-MNI152NLin2009cAsym_"
        f"atlas-{args.atlas}_desc-{args.parcel}_dseg.nii.gz"
    )
    ood_atlas_masker = NiftiLabelsMasker(
        labels_img=ood_atlas_path,
        standardize=False,
    )
    ood_atlas_masker.fit()

    # map Pearson correlations onto brain parcels
    nib.save(
        ood_atlas_masker.inverse_transform(
            np.array(results["parcelwise"]["test_R2"])
        ),
        f"{args.odir}/{args.participant}_dset-friends2OOD_{args.atlas}_"
        f"{args.parcel}_RidgeReg{modal_names}_R2_test_ood.nii.gz",
    )

    for k, v in results["parcelwise"]["test_R2_per_movie"].items():
        nib.save(
            ood_atlas_masker.inverse_transform(np.array(v)),
            f"{args.odir}/{args.participant}_dset-friends2OOD_{args.atlas}_"
            f"{args.parcel}_RidgeReg{modal_names}_R2_test_desc-{k}.nii.gz",
        )

    return


def test_ridgeReg(
    args: argparse.Namespace,
    R: RidgeCV,
    x_train: np.array,
    y_train: np.array,
    x_val: np.array,
    y_val: np.array,
    test_sets: dict,
) -> None:
    """.

    Exports RR results in .json file.
    """
    res_dict = {}

    # Global R2 scores
    res_dict["train_R2"] = R.score(x_train, y_train)
    res_dict["val_R2"] = R.score(x_val, y_val)

    y_test = np.concatenate([
        v["y_test"] for (k,v) in test_sets.items()], axis=0)
    x_test = np.concatenate([
        v["x_test"] for (k,v) in test_sets.items()], axis=0)
    res_dict["test_R2"] = R.score(x_test, y_test)
    res_dict["test_R2_per_movie"] = {
        k:R.score(v["x_test"], v["y_test"]) for (k,v) in test_sets.items()
    }


    # Parcel-wise predictions
    pred_train = R.predict(x_train)
    pred_val = R.predict(x_val)
    pred_test = R.predict(x_test)

    res_dict["parcelwise"] = {}
    res_dict["parcelwise"]["train_R2"] = (
        pearson_corr(y_train.T, pred_train.T)**2
    ).tolist()
    res_dict["parcelwise"]["val_R2"] = (
        pearson_corr(y_val.T, pred_val.T)**2
    ).tolist()
    res_dict["parcelwise"]["test_R2"] = (
        pearson_corr(y_test.T, pred_test.T)**2
    ).tolist()
    res_dict["parcelwise"]["test_R2_per_movie"] = {
        k: (pearson_corr(
            v["y_test"].T, R.predict(v["x_test"]).T
        )**2).tolist() for (k,v) in test_sets.items()
    }

    # export RR results
    Path(f"{args.odir}").mkdir(parents=True, exist_ok=True)
    m = ""
    for modal in args.modalities:
        m += f"_{modal}"
    if "text" in args.modalities:
        t_feat = {
            'text_pooled': 'BERTpool',
            'text_token': 'BERTtk',
            'text_OAI': 'OAIembs',
            'text_1hot': '1hot',
            'text_1hothtk': '1hothtk'
        }
        for feat in args.text_features:
            m += f"_{t_feat[feat]}"
    with open(f"{args.odir}/{args.participant}_dset-friends2OOD_ridgeReg{m}_{args.atlas}_{args.parcel}_result.json", 'w') as fp:
        json.dump(res_dict, fp)

    # export parcelwise scores as .nii images for visualization
    if args.bdir is not None:
        export_images(args, m, res_dict)


def main() -> None:
    """.

    This script performs a ridge regression on the Courtois Neuromod friends
    dataset.
    It uses multimodal features (visual / audio / text) extracted from the
    videos to predict parcellated BOLD time series.
    """
    args = get_arguments()

    print(vars(args))

    # Assign runs to train / validation sets
    train_grps, train_runs, val_runs, test_runs = split_episodes(args)

    # Build y matrices from BOLD timeseries
    y_train, length_train, train_groups = build_y(
        args, "friends", train_runs, train_grps,
    )
    y_val, length_val, _ = build_y(
        args, "friends", val_runs,
    )

    # Build X arrays from input features
    x_train = build_X(
        args, "friends", train_runs, length_train,
    )
    x_val = build_X(
        args, "friends", val_runs, length_val,
    )

    # Build OOD test set data per movie
    test_sets = {}
    for t in test_runs.keys():
        y_test, length_test, _ = build_y(args, "ood", test_runs[t])
        x_test = build_X(args, "ood", test_runs[t], length_test)
        test_sets[t] = {"y_test": y_test, "x_test": x_test}

    # Train ridge regression model on train set
    model = train_ridgeReg(
        x_train,
        y_train,
        train_groups,
        args.n_split,
    )

    # Test model and export performance metrics
    test_ridgeReg(
        args, model, x_train, y_train, x_val, y_val, test_sets,
    )


if __name__ == "__main__":
    sys.exit(main())
