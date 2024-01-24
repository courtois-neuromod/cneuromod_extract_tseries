Workflow Overview
=================

This library supports the extraction of denoised fMRI timeseries datasets from
the `Courtois Project on Neuronal Modelling (CNeuroMod) <https://www.cneuromod.ca/>`_.

The datasets are described `here <https://docs.cneuromod.ca/en/latest/DATASETS.html/>`_.

All CNeuroMod data are made available as a `DataLad collection on github <https://github.com/courtois-neuromod/>`_.
The dataset can be explored without downloading the data, and it is easy
to download only the subset of data needed for a project.

To request access to the datasets, please `apply here <https://www.cneuromod.ca/access/access/>`_.

The current code takes functional BOLD data preprocessed by fmri.prep as input, and
exports arrays of denoised timeseries as .HDF5 (one file per subject).

* :doc:`Install the code repository and dataset(s) <../installation.rst>`

* `test <https://github.com/courtois-neuromod/cneuromod_extract_tseries/blob/dev/docs/installation.rst/>`_

* `Create custom parcellations <https://github.com/courtois-neuromod/cneuromod_extract_tseries/blob/dev/docs/parcellation.rst/>`_

* `Run the code <https://github.com/courtois-neuromod/cneuromod_extract_tseries/blob/dev/docs/running.rst/>`_
