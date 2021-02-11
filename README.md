[![Build Status](https://travis-ci.org/neufieldrobotics/zernike_py.svg?branch=master)](https://travis-ci.org/neufieldrobotics/zernike_py)

# zernike_py

Python implementation of multi-scale Harris corner detector with zernike feature descriptor as described in ["Toward large-area mosaicing for underwater scientific applications"](https://bit.ly/3751KRP).  If you use zernike_py in academic work, please cite:

`O. Pizarro and H. Singh, "Toward large-area mosaicing for underwater scientific applications," in IEEE Journal of Oceanic Engineering, vol. 28, no. 4, pp. 651-672, Oct. 2003, doi: 10.1109/JOE.2003.819154.`

## Prerequisites
The following python packages are required to run this code:
### Required: 
  - opencv
  - numpy
### Optional:
  - matplotlib (for displaying reults and running demo.py)
  - scipy (when using like_matlab flag)
  - matlab_imresize (when using like_matlab flag included in repo)

## Installation
```sh
git clone https://gitlab.com/neufieldrobotics/zernike_py.git
cd zernike_py
# if opencv python is already installed
pip install -r requirements/base.txt
# if opencv python needs to be installed
pip install -r requirements/opencv_latest.txt
```

## Executing
```sh
python ./demo.py
```

