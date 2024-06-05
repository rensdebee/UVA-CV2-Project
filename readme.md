# DreamGaussian

This repository contains a refined version of [DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation](https://arxiv.org/abs/2309.16653), focussing on improving the text-to-3D pipeline.

Improvements made:
* Point-E / Shap-E Initialization option
* MVdream / Stable diffusion V2.1 ISM loss implemented
* SDS + ISM loss implemented for stage 2 texture optimization

## Example video: 
[![Example](https://markdown-videos-api.jorgenkh.no/url?url=https%3A%2F%2Fyoutu.be%2FrgkWRRVUFQE)](https://youtu.be/rgkWRRVUFQE)


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

# MVdream:
pip install git+https://github.com/bytedance/MVDream

# To use SuGaR, install the following:
pip install -r requirements_sugar.txt

```

Tested on:

- Ubuntu 22 with torch 1.12 & CUDA 11.6 on a V100.
- Windows 10 with torch 2.1 & CUDA 12.1 on a 3070.
- Red Hat 8.6 with torch 2.01 & CUDA 11.7 on a A100 (Snellius HPC).

## Usage

```bash
# Remove # comments before running

# Step 1
python main.py --config configs/text.yaml \ # Config file with hyper parameters
 prompt="<prompt>" \ # Prompt to create 3D object from
 point_e="<prompt>" \ # Prompt to intialize Point-E, use "SHAPE_<prompt>" to use Shap-E, remove to use random init
 mvdream=True \ # Boolean indicatin to use MVdream diffusion model, False uses Stable Diffusion V2.1
 stage1="SDS" \ # Stage 1 loss choose from MSE, SDS, ISM
 stage2="MSE" \ # Stage 2 loss choose from MSE, SDS, ISM
 outdir=<path> # Path to store object files
 
# Step 2
python main2.py --config configs/text.yaml \ # Config file with hyper parameters
 prompt="<prompt>" \ # Prompt to create 3D object from
 point_e="<prompt>" \ # Prompt to intialize Point-E, use "SHAPE_<prompt>" to use Shap-E, remove to use random init
 mvdream=True \ # Boolean indicatin to use MVdream diffusion model, False uses Stable Diffusion V2.1
 stage1="SDS" \ # Stage 1 loss choose from MSE, SDS, ISM
 stage2="MSE" \ # Stage 2 loss choose from MSE, SDS, ISM
 outdir=<path> # Path to store object files
 
 # Alternatively, step 2 with SuGaR refinement
 python sugar/run_sugar.py \
 --prompt "<prompt>"  \ 
 -c depth \ # condition type, depth used for this project
 -n {name} \ # project name, creates a directory using this name
 --resume # resumes a crashed step 2 attempt if available
```

```bash
# export all ./logs/*.obj to mp4 in ./videos
python scripts/convert_obj_to_video.py --dir ./logs
```

## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [threestudio](https://github.com/threestudio-project/threestudio)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [dearpygui](https://github.com/hoffstadt/DearPyGui)

The SuGaR implementation is largely adapted from the amazing official implementation of [Controllable Text-to-3D Generation via Surface-Aligned Gaussian Splatting](https://arxiv.org/abs/2403.09981).
The GitHub repository can be found [here](https://github.com/WU-CVGL/MVControl-threestudio).
