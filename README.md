## [CVPR 2024] Test-Time Training of Trajectory Prediction via Masked Autoencoder and Actor-specific Token Memory
Please follow below steps to run our code

## Installation
### Create virtual environment in Anaconda with env.yml
```
conda env create --file env.yaml -n t4p
conda activate t4p
```

### Install `T4P` (This repo)
```
git clone https://github.com/daeheepark/T4P
cd T4P
mkdir outputs && mkdir datasets
pip install -r requirements.txt
```

### Install a customized version of `trajdata`
```
git clone https://github.com/daeheepark/trajdata-t4p unified-av-data-loader
```
- Download raw datasets and follow the installation step of the above repo.
- The datasets should be located like this:
```
├── conf
├── ...
├── datasets
│   ├── nuScenes
│   ├── waymo
│   ├── interaction_single
│   ├── interaction_multi
│   └── lyft
├── train_test.py
└── train_ttt.py
```

## Acknowledgements
This repo is mostly built on top of [ForecastMAE](https://arxiv.org/pdf/2308.09882.pdf) and [trajdata](https://arxiv.org/pdf/2307.13924.pdf).
Thanks for their great works.

## Citation
If you found this repository useful, please consider citing our work:
```
@article{park2024t4p,
  title={T4P: Test-Time Training of Trajectory Prediction via Masked Autoencoder and Actor-specific Token Memory},
  author={Park, Daehee and Jeong, Jaeseok and Yoon, Sung-Hoon and Jeong, Jaewoo and Yoon, Kuk-Jin},
  journal={arXiv preprint arXiv:2403.10052},
  year={2024}
}
```