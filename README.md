
---
# Estimating entropy production in a stochastic system with odd-parity variables

[![arxiv](http://img.shields.io/badge/arXiv-2112.04681-B31B1B.svg)](https://arxiv.org/abs/2112.04681)
[![LICENSE](https://img.shields.io/github/license/kdkyum/odd_neep.svg)](https://github.com/kdkyum/odd_neep/blob/main/LICENSE)

Authors: Dong-Kyum Kim<sup>1*</sup>, Sangyun Lee<sup>2*</sup>, and Hawoong Jeong<sup>1,3</sup><br>
<sub>\* Equal contribution</sub>

<sup>1</sup> <sub>Department of Physics, KAIST</sub>
<sup>2</sup> <sub>School of Physics, KIAS</sub>
<sup>3</sup> <sub>Center for Complex Systems, KAIST</sub>

## Introduction

This repo contains source code for the runs in [Estimating entropy production in a stochastic system with odd-parity variables](https://arxiv.org/abs/2112.04681)

## Installation

Supported platforms: MacOS and Ubuntu, Python 3.7

Installation using [Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html):

```bash
git clone https://github.com/kdkyum/odd_neep.git
cd odd_neep
conda create -y --name odd_neep python=3.7
conda activate odd_neep
pip install -r requirements.txt
```

To enable gpu usage, install gpu version `torch` package from [PyTorch](https://pytorch.org).  

## Usage

* Training for Underdamped bead-spring model.

```bash
python main_ubs.py \
  --save results/ubs \
  --n_layer 2 \
  --n_hidden 256 \
  --N 2 \
  --Tc 1 \
  --Th 10 \
  --m 0.01 \
  --lr 1e-5 \
  --wd 0 \
  --dropout 0 \
  --trj_num 10000 \
  --trj_len 4000 \
  --record_freq 400 \
  --n_iter 100000 \
  --seed 42
```

* Training for odd-parity Markov jump process.

```bash
python main_omj.py \
  --save results/odd_markov_jump/c10 \
  --n_layer 2 \
  --n_hidden 256 \
  --trj_len 10000 \
  --trj_num 50 \
  --N 10 \
  --c 10 \
  --lr 1e-5 \
  --n_iter 10000 \
  --record_freq 500 \
  --batch_size 4096 \
  --seed 42 
```

## Bibtex
Cite the following Bibtex.
```bibtex
@article{kim2021odd_neep,
  title={Learning entropy production via neural networks},
  author={Dong-Kyum Kim and Sangyun Lee and Hawoong Jeong},
  journal={arXiv preprint arXiv:2112.04681},
}
```

## License

This project following the MIT license.
