# DreamGaussian

This repository contains a refined version of [DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation](https://arxiv.org/abs/2309.16653), improving the text-to-3D pipeline

### [Project Page](https://dreamgaussian.github.io) | [Arxiv](https://arxiv.org/abs/2309.16653)


## Install

```bash
pip install -r requirements.txt

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# simple-knn
pip install ./simple-knn

# nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast/

# kiuikit
pip install git+https://github.com/ashawkey/kiuikit

# To use MVdream, also install:
pip install git+https://github.com/bytedance/MVDream

# To use ImageDream, also install:
pip install git+https://github.com/bytedance/ImageDream/#subdirectory=extern/ImageDream
```

Tested on:

- Ubuntu 22 with torch 1.12 & CUDA 11.6 on a V100.
- Windows 10 with torch 2.1 & CUDA 12.1 on a 3070.

## Usage

I
## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [threestudio](https://github.com/threestudio-project/threestudio)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [dearpygui](https://github.com/hoffstadt/DearPyGui)
