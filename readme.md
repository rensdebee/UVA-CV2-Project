# DreamGaussian

This repository contains the unofficial implementation for [DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation](https://arxiv.org/abs/2309.16653).


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

# To use SuGaR, install the following:
pip install -r requirements_sugar.txt
```

Tested on:

- Ubuntu 22 with torch 1.12 & CUDA 11.6 on a V100.
- Windows 10 with torch 2.1 & CUDA 12.1 on a 3070.
- Red Hat 8.6 with torch 2.01 & CUDA 11.7 on a A100 (Snellius HPC).

## Usage
Text-to-3D:

```bash
### training gaussian stage
python main.py --config configs/text.yaml prompt="a photo of an icecream" save_path=icecream

### training mesh stage
python main2.py --config configs/text.yaml prompt="a photo of an icecream" save_path=icecream

### refining gaussians with SuGaR
python sugar/run_sugar.py --prompt {prompt} -c depth -n {name} --resume
```

Please check `./configs/text.yaml` for more options.


Helper scripts:

```bash
# run all image samples (*_rgba.png) in ./data
python scripts/runall.py --dir ./data --gpu 0

# run all text samples (hardcoded in runall_sd.py)
python scripts/runall_sd.py --gpu 0

# export all ./logs/*.obj to mp4 in ./videos
python scripts/convert_obj_to_video.py --dir ./logs
```

## Tips
* The world & camera coordinate system is the same as OpenGL:
```
    World            Camera        
  
     +y              up  target                                              
     |               |  /                                            
     |               | /                                                
     |______+x       |/______right                                      
    /                /         
   /                /          
  /                /           
 +z               forward           

elevation: in (-90, 90), from +y to -y is (-90, 90)
azimuth: in (-180, 180), from +z to +x is (0, 90)
```

* Trouble shooting OpenGL errors (e.g., `[F glutil.cpp:338] eglInitialize() failed`): 
```bash
# either try to install OpenGL correctly (usually installed with the Nvidia driver), or use force_cuda_rast:
python main.py --config configs/image_sai.yaml input=data/name_rgba.png save_path=name force_cuda_rast=True

kire mesh.obj --force_cuda_rast
```

## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [threestudio](https://github.com/threestudio-project/threestudio)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [dearpygui](https://github.com/hoffstadt/DearPyGui)

## Citation

```
@article{tang2023dreamgaussian,
  title={DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation},
  author={Tang, Jiaxiang and Ren, Jiawei and Zhou, Hang and Liu, Ziwei and Zeng, Gang},
  journal={arXiv preprint arXiv:2309.16653},
  year={2023}
}
```
