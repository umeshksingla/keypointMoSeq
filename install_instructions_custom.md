# How to create a custom conda environment with working copies of sleap and keypoint_moseq and GPU support

## Package versions (important for compatibility)
	- OS
	- cudatoolkit
	- cudnn
	- python
	- tensorflow
	- sleap
	- jax/jaxlib
	- keypoint_moseq

## Installation steps
1. First create a conda environment with the appropriate python, cudatoolkit and cudnn versions. 
`conda create -n keypoint_moseq_sleap -c conda-forge cudatoolkit cudnn python=3.8` 
2. Activate this conda environment. 
3. Install sleap (with pip) because conda installing sleap doesn't support python > 3.7 and jax doesn't support python < 3.7. 
`pip install sleap`
4. Activate a python terminal. Import sleap. Verify that sleap is installed and can see the GPU. Optionally, verify that tensorflow is installed and can also see the GPU.
`
>>> import sleap
>>> sleap.versions()
SLEAP: 1.2.9
TensorFlow: 2.8.4
Numpy: 1.22.4
Python: 3.8.15
OS: Windows-10-10.0.19044-SP0
>>> sleap.system_summary()
GPUs: 1/1 available
  Device: /physical_device:GPU:0
         Available: True
        Initalized: True
     Memory growth: None
>>> exit()
`
5. Do `conda list` to verify that you have cudatoolkit > 11.4 and cudnn > 8.2. 
6. Find the appropriate jaxlib whl here: https://whls.blob.core.windows.net/unstable/index.html
7. Install jax: ` pip install jax https://whls.blob.core.windows.net/unstable/cuda111/jaxlib-0.3.22+cuda11.cudnn82-cp38-cp38-win_amd64.whl`
8. Verify that jax is installed and can see the GPU: Open a python terminal, import jax and use `jax.devices()` 
`
>>> import jax
>>> jax.devices()
[StreamExecutorGpuDevice(id=0, process_index=0)]
>>> exit()
`
9. Now install keypointMoseq `pip install -e keypointMoseq`
10. et Voila! 


## Useful links and references
1. https://sleap.ai/installation.html (pip installation)
2. https://github.com/calebweinreb/keypointMoSeq/tree/user_friendly_pipeline (general outline of steps to install keypointMoseq but don't follow these steps verbatim)
3. https://github.com/google/jax#installation
4. https://github.com/cloudhan/jax-windows-builder
5. https://askubuntu.com/questions/1338314/setting-up-tensorflow-gpu-conda-environment-with-cuda-11-2-and-cudnn-8-1-8-2-cu 
