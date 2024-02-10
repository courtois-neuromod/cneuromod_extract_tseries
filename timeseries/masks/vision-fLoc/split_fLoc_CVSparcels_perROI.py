import glob, os, sys
import nibabel as nib
import nilearn
import numpy as np

import argparse


def get_arguments():

    parser = argparse.ArgumentParser(description="Exports probabilistic masks in cvs_avg53 space for Kanwisher parcels")
    parser.add_argument('--in_dir', default='', type=str, help='absolute path to input directory')
    parser.add_argument('--out_dir', default='', type=str, help='absolute path to output directory')

    args = parser.parse_args()

    return args


def split_rois(in_dir, out_dir):
    '''
    Script takes Kanwisher group parcels from fLoc contrasts in cvs_avg35 space
    and generates separate masks per ROI
    ROIs include:
    - face contrast: FFA (fusiform face area), OFA (occipital face area),
                     pSTS (posterior superrior temporal sulcus)
    - body contrast: EBA (extrastriate body area)
    - scene contrast: PPA (parahippocampal place area), MPA (medial place area / RSP),
                      OPA (occipital place area)

    Parcel numbers were selected based on ROIs shown on flatmaps in the
    THINGS-data paper.
    In that paper, ROIs included: V1, V2 and V3 from retinotopy
    From fLoc
    face:
    - FFA (fusiform face area; L: 14 blue, R: 5 purple),
    - OFA (occipital face area; L:15 beige, R: 4 red)
    - pSTS (posterior sup temporal sulcus; L:7 green, R: 2 blue)
    body:
    - EBA (extrastriate body area, L: 2 blue, R: 3 white)
    scene:
    - PPA (parahippocampal place area, L: 4 red, R: 5 purple),
    - MPA (medial place area / RSP; L: 7 green, R: 6 mauve),
    - OPA (occipital place area, L: 8 light green, R: 9 orange)
    object:
    - LOC (lateral occipital cortex)

    ROI id numbers were identified by looking at the parcel positions within the
    THINGS-data flat maps, as well as their position on surfaces visualized with
    Freesurfer and descriptions from the Kanwisher group's methods paper.
    https://web.mit.edu/bcs/nklab/media/pdfs/julian.neuroimage.2012.pdf

    Note: to visualize cvs parcels in freesurfer, type freeview to open the GUI.
    Load volume cvs_avg35 template, e.g. from /usr/local/freesurfer/7.3.2/subjects/csv_avg35/mri/T1.mgz
    Load volume parcel: e.g., cvs_scene_parcels/fROIs-fwhm-5-0.0001.nii
    Select "Lookup Table" as the color map, then load the parcel file's accompanying "parcel_LUT.txt" file as the Lookup table ("load lookup table")
    '''

    roi_number_dict = {
        'face': {
            'FFA': {
                'L': 14.0,
                'R': 5.0
            },
            'OFA': {
                'L': 15.0,
                'R': 4.0
            },
            'pSTS': {
                'L': 7.0,
                'R': 2.0
            }
        },
        'body': {
            'EBA': {
                'L': 2.0,
                'R': 3.0
            }
        },
        'scene': {
            'PPA': {
                'L': 4.0,
                'R': 5.0
            },
            'MPA': {
                'L': 7.0,
                'R': 6.0
            },
            'OPA': {
                'L': 8.0,
                'R': 9.0
            }
        }
    }

    contrast_list = list(roi_number_dict.keys())

    for c in contrast_list:
        c_file = nib.load(
            f"{in_dir}/cvs_{c}_parcels/cvs_{c}_parcels/fROIs-fwhm_5-0.0001.nii"
        )
        c_affine = c_file.affine
        c_array = c_file.get_fdata()

        ROI_list = list(roi_number_dict[c].keys())

        for roi in ROI_list:
            left_val = roi_number_dict[c][roi]['L']
            right_val = roi_number_dict[c][roi]['R']

            left_array = (c_array == left_val)#.astype(float)
            right_array = (c_array == right_val)#.astype(float)
            bilat_array = (left_array + right_array).astype(bool)

            left_mask = nib.nifti1.Nifti1Image(
                left_array.astype(float), affine=c_affine
            )
            right_mask = nib.nifti1.Nifti1Image(
                right_array.astype(float), affine=c_affine
            )
            bilat_mask = nib.nifti1.Nifti1Image(
                bilat_array.astype(float), affine=c_affine
            )

            nib.save(left_mask, f'{out_dir}/tpl-CVSavg35_atlas-vision-fLoc-kanwisher_desc-{c}-{roi}-L_mask.nii')
            nib.save(right_mask, f'{out_dir}/tpl-CVSavg35_atlas-vision-fLoc-kanwisher_desc-{c}-{roi}-R_mask.nii')
            nib.save(bilat_mask, f'{out_dir}/tpl-CVSavg35_atlas-vision-fLoc-kanwisher_desc-{c}-{roi}_mask.nii')


if __name__ == '__main__':

    args = get_arguments()

    in_dir = args.in_dir
    out_dir = args.out_dir

    split_rois(in_dir, out_dir)
