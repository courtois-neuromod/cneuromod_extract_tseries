

import nibabel as nib
import numpy as np

mask_path = "/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/masks/gm-masks/freesurfer"

for s in range(1, 7):
    snum = str(s).zfill(2)

    mask_FS_aseg = nib.load(f"{mask_path}/sub-{snum}_aseg.nii.gz")

    print(np.unique(mask_FS_aseg.get_fdata()))

    # Indices from here https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
    keep_list = [
        3.0,    # 3   Left-Cerebral-Cortex
        8.0,    # 8   Left-Cerebellum-Cortex
        10.0,   # 10  Left-Thalamus-Proper*
        11.0,   # 11  Left-Caudate
        12.0,   # 12  Left-Putamen
        13.0,   # 13  Left-Pallidum
        17.0,   # 17  Left-Hippocampus
        18.0,   # 18  Left-Amygdala
        26.0,   # 26  Left-Accumbens-area
        28.0,   # 28  Left-VentralDC
        42.0,   # 42  Right-Cerebral-Cortex
        47.0,   # 47  Right-Cerebellum-Cortex
        49.0,   # 49  Right-Thalamus-Proper*
        50.0,   # 50  Right-Caudate
        51.0,   # 51  Right-Putamen
        52.0,   # 52  Right-Pallidum
        53.0,   # 53  Right-Hippocampus
        54.0,   # 54  Right-Amygdala
        58.0,   # 58  Right-Accumbens-area
        60.0    # 60  Right-VentralDC
    ]

    flat_vals = (mask_FS_aseg.get_fdata().reshape([-1])).tolist()
    mask_gm_aseg = np.array([x in keep_list for x in flat_vals]).reshape(mask_FS_aseg.shape).astype(int)

    # Sanity check
    m_array = mask_FS_aseg.get_fdata()
    assert np.sum(np.array((m_array.reshape([-1])).tolist()).reshape(m_array.shape) == m_array) == m_array.size

    mask_nii = nib.nifti1.Nifti1Image(
        mask_gm_aseg,
        affine=mask_FS_aseg.affine,
        dtype="uint8",
    )
    nib.save(mask_nii, f"{mask_path}/sub-{snum}_space-T1w_label-GM_dseg.nii.gz")

    mask_nii_float = nib.nifti1.Nifti1Image(
        mask_gm_aseg,
        affine=mask_FS_aseg.affine,
        dtype=np.float64,
    )
    nib.save(mask_nii_float, f"{mask_path}/sub-{snum}_space-T1w_label-GM_desc-float_dseg.nii.gz")
