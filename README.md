# Multi-label Classification with Deep Learning

This is a Python3.5 version and keras 2.2.0 implementation for the paper: ``[ADIOS: Architectures Deep In Output Space](http://bengio.abracadoudou.com/cv/publications/pdf/cisse_2016_icml.pdf)"

`adios.utils.assemble.assemble` helper function provides and handy way to construct ADIOS and MLP models from config dictionaries.

All example scripts are given in `scripts/`.

**Note:** `keras.models.Graph` was no longer supported starting from `keras-v1.0` as of April, 2016. The last version of ADIOS used the legacy code, `keras.legacy.models.Graph`, but this has issues with new Keras. Thus we replaced it with `Model`.


### Requirements
- `scikit-multilearn`
- `NumPy`
- `pyyaml`
- `Theano`
- `keras==2.2.0`
- `scikit-learn`

The requirements can be installed via `pip` as follows:

```bash
$ pip install -r requirements.txt
```

Optional (needed only for using Jobman):
- `argparse`
- [Jobman](http://deeplearning.net/software/jobman/about.html)


### Installation
To use the code, we recommend installing it as Python package in the development mode as follows:

```bash
$ python setup.py develop [--user]
```

The `--user` flag (optional) will install the package for a given user only.

This version shows an example of the Delicious dataset, following the MULAN format. You may also need to switch the scikit-multilearn arff reader to non-binary, so that you parse the datasets.

### Citation policy
If you use this code (in full or in part) for academic purposes, please cite the original ADIOS paper.
