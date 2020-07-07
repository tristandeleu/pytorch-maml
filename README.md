# Model-Agnostic Meta-Learning
[![Documentation](https://img.shields.io/badge/1.5-PyTorch-EE4C2C)](https://pytorch.org/)

An implementation of Model-Agnostic Meta-Learning (MAML) in [PyTorch](https://pytorch.org/) with [Torchmeta](https://github.com/tristandeleu/pytorch-meta).

### Getting started
To avoid any conflict with your existing Python setup, it is suggested to work in a virtual environment with [`virtualenv`](https://docs.python-guide.org/dev/virtualenvs/). To install `virtualenv`:
```bash
pip install --upgrade virtualenv
```
Create a virtual environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Requirements
 - Python 3.6 or above
 - PyTorch 1.5
 - Torchvision 0.6
 - Torchmeta 1.4.6

### Usage
You can use [`train.py`](train.py) to meta-train your model with MAML. For example, to run MAML on Omniglot 1-shot 5-way with default parameters from the original paper:
```bash
python train.py /path/to/data --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 600 --output-folder /path/to/results
```
The meta-training script creates a configuration file you can use to meta-test your model. You can use [`test.py`](test.py) to meta-test your model:
```bash
python test.py /path/to/results/config.json
```

### References
The code available in this repository is mainly based on the paper
> Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep
networks. _International Conference on Machine Learning (ICML)_, 2017 [[ArXiv](https://arxiv.org/abs/1703.03400)]

If you want to cite this paper
```
@article{finn17maml,
  author    = {Chelsea Finn and Pieter Abbeel and Sergey Levine},
  title     = {Model-{A}gnostic {M}eta-{L}earning for {F}ast {A}daptation of {D}eep {N}etworks},
  journal   = {International Conference on Machine Learning (ICML)},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.03400}
}
```
