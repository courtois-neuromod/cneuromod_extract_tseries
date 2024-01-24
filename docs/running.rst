Running the code
================

The code to extract timeseries from a given CNeuroMod dataset can be run with
a simple command line that specifies the dataset and the extraction parameters.

E.g.,
::
    python run.py dataset=shinobi parcellation=mist444

Under the hood, the code relies on a combination of .yaml config files.
We use `Hydra <https://hydra.cc/>`_ to flexibly combine datasets, parcellation
atlases and denoising strategies.

Users can call the existing config files from the command line, or create their
own as needed. Depending on the analysis specified, supporting documents
(e.g., grey matter masks, parcellation atlases) may need to be added, as
detailed below.

The parameters to specify are the following.


1. Dataset
----------

Dataset config files are saved under
``cneuromod_extract_tseries/timeseries/config/dataset/<dset_name>.yaml``

When launching the script, you must specify the name of an input dataset.
The name must correspond to one of the ``<dset_name>.yaml`` files.
E.g.,
::
    python run.py dataset=movie10 parcellation=mist444

The script will look for the input data under
``<data_dir>/<dset_name>.fmriprep``

By default, ``<data_dir>`` corresponds to ``cneuromod_extract_tseries/data``.
This default can be overriden at the command line to match another dataset location.
E.g.,
::
    python run.py dataset=movie10 data_dir=/home/user/project/my_data_dir parcellation=mist444

Alternatively, the ``data_dir`` and ``dset_name`` variables can be modified
directly in a dataset ``config/dataset/<dset_name>.yaml`` file to reflect the data location.
