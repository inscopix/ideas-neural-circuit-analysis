# Neural Stats

This container has common python packages for neural statistical analysis.

**Base Image:** [python3.13.5](https://hub.docker.com/layers/library/python/3.13.5/images/sha256-4ddc936bb22ccb7af9355ed648625db58ca896dd5f40d5fcdbe53b0d7b33ebf5)

## Packages Installed:

In order to facilitate neural processing, this container image consists of many python packages commonly used in neural analysis tools.

### Stats

For statistical analysis, the following packages are installed:

* `numpy==2.2.6`
* `pandas==2.3.3`
* `scipy==1.16.3`
* `scikit-learn==1.7.2`
* `scikit-image==0.25.2`
* `pingouin==0.5.5`
* `opencv-python==4.12.0.88`
* `statsmodels==0.14.5`

### Plotting

For plotting, the following packages are installed, which can be used to generate figures for result files: 
* `matplotlib==3.10.7`
* `plotly==6.5.0`
* `seaborn==0.13.2`
* `bokeh==3.8.1`

### I/O

For i/o and data processing operations, the following packages are installed:

* `pyarrow==22.0.0`
* `h5py==3.15.1`
* `tifffile==2025.10.16`
* `ImageIO==2.37.2`
* `imageio-ffmpeg==0.6.0`
* `pillow==12.0.0`

### Neuro

Additionally, the following packages are installed for neural processing specifically: 

* `ideas-python==1.0.0`
* `isx==2.1.0`
* `pynwb==3.1.2`
* `pynapple==0.10.1`

### Development

For development, the following packages are installed:

* `beartype==0.22.6`
* `pytest==9.0.1`
