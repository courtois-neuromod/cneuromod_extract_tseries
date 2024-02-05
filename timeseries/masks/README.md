Code to produce ROI masks in single-subject space,
or single-subject masks in MNI space from functional data (not from template)


Vision: localisers in 3 out of 6 subjects for floc and retino in T1w space
Use group atlases for other subjects?

Audio: Maelle used MIST (MNI space)

Language: ask Valentina and Maria
some from HCPtrt task... Federenko localizers


Functional networks
FP code: in MNI space
derive single-subject networks that can be converted into binary masks
based on seed coordinates from Yeo's 7 functional networks

Links:
- https://github.com/courtois-neuromod/fmri-generator/blob/master/scripts/seed_connectivity.py
- https://nilearn.github.io/stable/auto_examples/03_connectivity/plot_seed_to_voxel_correlation.html
"The maps presented in the results are using seeds in 6 of the 7 Yeo networks [30]: the default-mode network with a seed in the cingulate gyrus (-7, -52, 26), the visual network with a seed in the superior lingual gyrus (-16, -74, 7), the sensorimotor network with a seed in the postcentral gyrus (-41, -20, 62), the dorsal attentional network with another seed in the postcentral gyrus (-34, -38, 44), the ventral attentional network with a seed in the cingulate gyrus (-5, 15, 32) and the fronts-parietal network with a seed in the middle frontal gyrus (-40, 50, 7). The seventh limbic network was excluded because it is composed of regions with high signal loss and distortions due to field inhomogeneity."





converting MNI coordinates (in mm) to T1W coordinates (in voxels)

With ANTS: antsApplyTransformsToPoints
https://neurostars.org/t/extracting-individual-transforms-from-composite-h5-files-fmriprep/2215/14
https://antspy.readthedocs.io/en/latest/_modules/ants/registration/apply_transforms.html
https://github.com/stnava/chicken/blob/master/runthis.sh#L53-L54
https://github.com/stnava/chicken/blob/d3e3855b9c13f8096aed27a0addf66b1a11e6bec/runthis.sh#L53-L54

Coordinate systems:
https://www.slicer.org/wiki/Coordinate_systems


USE FSL FLIRT std2imgcoord, combine two transformations...

https://neurostars.org/t/coordinates-from-mni-to-subject-space-using-transform-h5-file/5994/3

https://neurostars.org/t/mni-coordinates-to-subjects-epi/5628

https://neurostars.org/t/mni-to-native-space/2107/2

https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT/UserGuide#img2imgcoord
