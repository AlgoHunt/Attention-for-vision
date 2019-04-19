# Attention-for-vision
an off-the-shelf cuda implementation of various strange and not so strange way of doing attention on 4d tensor

## Usage

As this implementations utilize JIT Compiler, you can simply load a module as import xx_weight form xx, and the c++/cuda code
will be build automaticly. see function.py in each package for better understanding.

## Requirements

To install PyTorch, please refer to https://github.com/pytorch/pytorch#installation.

**NOTE: this implementation is  _adapted_ for PyTorch v1.0**.

To install all dependencies using pip, just run:
```bash
pip install -r requirements.txt
```

Most of the attention have native CUDA implementations, which are compiled using Pytorch v1.0's newly introduced
extension mechanism, which requires a package called `ninja`.
This can easy be installed from most distributions' package managers, _e.g._ in Ubuntu derivatives:
```bash
sudo apt-get install ninja-build
```
In case PyTorch is installed via conda, `ninja` will be automatically installed too.