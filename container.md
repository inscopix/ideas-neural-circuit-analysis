# Neural Stats

This container has common python packages for neural statistical analysis.

**Base Image:** [python3.13.5](https://hub.docker.com/layers/library/python/3.13.5/images/sha256-4ddc936bb22ccb7af9355ed648625db58ca896dd5f40d5fcdbe53b0d7b33ebf5)

## Packages Installed:

In order to facilitate neural processing, this container image consists of many python packages commonly used in neural analysis tools.

### Stats

For statistical analysis, the following packages are installed:

* `numpy`
* `pandas`
* `scipy`
* `scikit-learn`
* `scikit-image`
* `pingouin`
* `opencv-python`
* `statsmodels`

### Plotting

For plotting, the following packages are installed, which can be used to generate figures for result files: 
* `matplotlib`
* `plotly`
* `seaborn`
* `bokeh`

### I/O

For i/o and data processing operations, the following packages are installed:

* `pyarrow`
* `h5py`
* `tifffile`
* `imageio`
* `imageio-ffmpeg`
* `Pillow`

### Neuro

Additionally, the following packages are installed for neural processing specifically: 

* `ideas-python`
* `isx`
* `pynwb`
* `pynapple`

### Test

For unit testing, the following packages are installed:

* `pytest`
