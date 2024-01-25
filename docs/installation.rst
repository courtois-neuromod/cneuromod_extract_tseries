Installation
============

All CNeuroMod data are made available as a `DataLad collection on github <https://github.com/courtois-neuromod/>`_.
The released datasets are described `here <https://docs.cneuromod.ca/en/latest/DATASETS.html>`_.
They can be explored without downloading the data, and it is easy
to download only the subset of data needed for a project.


1. Requesting access
--------------------

You can apply for access to the CNeuroMod datasets `here <https://www.cneuromod.ca/access/access/>`_.

You will receive login credentials to access the NeuroMod git and the
NeuroMod Amazon S3 fileserver so you can download the data.
`See here <https://docs.cneuromod.ca/en/latest/ACCESS.html#downloading-the-dataset/>`_ for additional information on accessing the data.


2. Installing DataLad
---------------------

Install a recent version of the `Datalad software <https://www.datalad.org/>`_,
a tool for versioning large data structures in a git repository available
for Linux, OSX and Windows.

If not already present, we also recommend creating an SSH key on the machine
where the dataset will be installed and adding it to Github. See the
`official github instructions <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account/>`_ on how to create and add a key to your account.


3. Installing the code repository
---------------------------------

Clone the current code repository from GitHub.
.. code-block:: bash
  datalad clone git@github.com:courtois-neuromod/cneuromod_extract_tseries.git
  cd cneuromod_extract_tseries


4. Installing the dataset(s)
----------------------------
Specify your CNeuroMod login credentials as environment variables in your
``bash`` console to download data from the S3 file server.

Use the **access_key** and **secret_key** you received when granted access
to the dataset.
::
  export AWS_ACCESS_KEY_ID=<s3_access_key>  AWS_SECRET_ACCESS_KEY=<s3_secret_key>

Install the dataset repositery as ``<dataset_name>.fmriprep``.
The default location is within ``cneuromod_extract_tseries/data``

E.g.,
:: 
  cd data
  datalad install git@github.com:courtois-neuromod/friends.fmriprep.git
  cd friends.fmriprep

By default, the latest stable (recommended) release will be installed.
If you need another version (e.g., to reproduce a result), you can switch
to the appropriate tag/branch.
::
  git checkout rel/2022

Pull the dataset repository's data from the server.
To download the entire dataset recursively, do
::
  datalad get -r *

To download a single subject's preprocessed data (e.g., sub-01)
::
  datalad get -r sub-01/*


5. Setting up the virtual environment
-------------------------------------
Install the required libraries within a virtual environment.
::
  pip install -r requirements.txt



TODO : add alternative instructions to run code inside aptainer image
