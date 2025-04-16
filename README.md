**[summary](#summary) | [setup](#setup) | [resources](#resources) | [license](#license)**
# 2025-sustech-casing-demo

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ubcgif/2025-sustech-casing-demo/main)
[![License](https://img.shields.io/github/license/ubcgif/2025-sustech-casing-demo.svg)](https://github.com/ubcgif/2025-sustech-casing-demo/blob/main/LICENSE)

## Summary

These notebooks show an example of using the 3D cylindrical code in SimPEG to simulate EM experiments with a vertical, steel-cased well. 


## Setup

There are a few things you'll need to use this app on your own computer:

1. A working Python installation. I recommend using [Miniforge](https://github.com/conda-forge/miniforge), but you can also use [Anaconda](https://www.anaconda.com/download)
2. To install the [conda environment](./environment.yml)
3. A web browser that works with Jupyter
   (basically anything except Internet Explorer)

Alternatively, you can run this notebook on the cloud with binder: https://mybinder.org/v2/gh/ubcgif/2025-sustech-casing-demo/main

**Windows users:** If you are using Anaconda, when you see "*terminal*" in the instructions,
this means the "*Anaconda Prompt*" program for you.

### Step 1: Python

Install Python on your machine. I recommend using [Miniforge](https://github.com/conda-forge/miniforge), but you can also use [Anaconda](https://www.anaconda.com/download)

If you are looking for tutorials, you can take a look at these videos:
for [Windows](https://youtu.be/FdatS_NKVrM)
and [Linux](https://youtu.be/3ncwbHyZeAg)

This will get you a working Python 3 installation with the `conda` package
manager. If you already have one, you can skip this step.

### Step 2: Download the notebook and code

To access the notebooks, there are 3 options (in order of preference):
1. You can download a zip file containing https://github.com/ubcgif/2025-sustech-casing-demo/archive/refs/heads/main.zip.
2. You can run the notebooks online with binder through: https://mybinder.org/v2/gh/ubcgif/2025-sustech-casing-demo/main

If you download the zip file, unzip it, and then open up a terminal in the `2025-sustech-casing-demo` directory.

### Step 3: Create the conda environment

With an open terminal in the `2025-casing-demo` directory, create the `2025-casing-demo` conda environment using the following. If you have a recent Mac with an M1 or M2 chip replace `environment.yml` with `environment-mac.yml`:
```
conda env create -f environment.yml
```
and activate the environment
```
conda activate 2025-casing-demo
```

### Step 4: Launching the notebooks

Once you have activated the conda environment, you can launch the notebooks
```
jupyter lab
```
Jupyter will then launch in your web browser.

## Resources

**Resources on SimPEG**
- [Docs](http://docs.simpeg.xyz/)
- [Mattermost chat](https://mattermost.softwareunderground.org/simpeg)

## License

All code and text in this repository is free software: you can redistribute it and/or
modify it under the terms of the MIT License.
A copy of this license is provided in [LICENSE](LICENSE).
