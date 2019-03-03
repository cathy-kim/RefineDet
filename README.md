# Single-Shot Refinement Neural Network for Object Detection(RefineDet) NVCaffe ver. 

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

By [Shifeng Zhang](http://www.cbsr.ia.ac.cn/users/sfzhang/), [Longyin Wen](http://www.cbsr.ia.ac.cn/users/lywen/), [Xiao Bian](https://sites.google.com/site/cvbian/), [Zhen Lei](http://www.cbsr.ia.ac.cn/users/zlei/), [Stan Z. Li](http://www.cbsr.ia.ac.cn/users/szli/). 

We propose a novel single-shot based detector, called RefineDet, that achieves better accuracy than two-stage methods and maintains comparable efficiency of one-stage methods. You can use the code to train/evaluate the RefineDet method for object detection. For more details, please refer to our [paper](https://arxiv.org/pdf/1711.06897.pdf).

<p align="left">
<img src="https://github.com/sfzhang15/RefineDet/blob/master/refinedet_structure.jpg" alt="RefineDet Structure" width="777px">
</p>

| System | VOC2007 test *mAP* | **FPS** (Titan X) | Number of Boxes | Input resolution
|:-------|:-----:|:-------:|:-------:|:-------:|
| [Faster R-CNN (VGG16)](https://github.com/ShaoqingRen/faster_rcnn) | 73.2 | 7 | ~6000 | ~1000 x 600 |
| [YOLO (GoogLeNet)](http://pjreddie.com/darknet/yolo/) | 63.4 | 45 | 98 | 448 x 448 |
| [YOLOv2 (Darknet-19)](http://pjreddie.com/darknet/yolo/) | 78.6 | 40 | 1445 | 544 x 544 |
| [SSD300* (VGG16)](https://github.com/weiliu89/caffe/tree/ssd) | 77.2 | 46 | 8732 | 300 x 300 |
| [SSD512* (VGG16)](https://github.com/weiliu89/caffe/tree/ssd) | 79.8 | 19 | 24564 | 512 x 512 |
| RefineDet320 (VGG16) | 80.0 | 40 | 6375 | 320 x 320 |
| RefineDet512 (VGG16) | **81.8** | 24 | 16320 | 512 x 512 |


<p align="left">
<img src="https://github.com/sfzhang15/RefineDet/blob/master/refinedet_results.jpg" alt="RefineDet results on multiple datasets" width="770px">
</p>


# NVCaffe

NVIDIA Caffe ([NVIDIA Corporation &copy;2017](http://nvidia.com)) is an NVIDIA-maintained fork
of BVLC Caffe tuned for NVIDIA GPUs, particularly in multi-GPU configurations.
Here are the major features:
* **16 bit (half) floating point train and inference support**.
* **Mixed-precision support**. It allows to store and/or compute data in either 
64, 32 or 16 bit formats. Precision can be defined for every layer (forward and 
backward passes might be different too), or it can be set for the whole Net.
* **Layer-wise Adaptive Rate Control (LARC) and adaptive global gradient scaler** for better
 accuracy, especially in 16-bit training.
* **Integration with  [cuDNN](https://developer.nvidia.com/cudnn) v7**.
* **Automatic selection of the best cuDNN convolution algorithm**.
* **Integration with v2.2 (or higher) of [NCCL library](https://github.com/NVIDIA/nccl)**
 for improved multi-GPU scaling.
* **Optimized GPU memory management** for data and parameters storage, I/O buffers 
and workspace for convolutional layers.
* **Parallel data parser, transformer and image reader** for improved I/O performance.
* **Parallel back propagation and gradient reduction** on multi-GPU systems.
* **Fast solvers implementation with fused CUDA kernels for weights and history update**.
* **Multi-GPU test phase** for even memory load across multiple GPUs.
* **Backward compatibility with BVLC Caffe and NVCaffe 0.15 and higher**.
* **Extended set of optimized models** (including 16 bit floating point examples).
* _Experimental feature (no official support)_ **Multi-node training** (since v0.17.1, NCCL 2.2 and OpenMPI 2 required).
* _Experimental feature (no official support)_ **TRTLayer** (since v0.17.1, can be used as inference plugin).

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
   
## Contributions

Please read, sign and attach enclosed agreement **NVIDIA_CLA_v1.0.1.docx**
to your PR.

## Useful notes

Libturbojpeg library is used since 0.16.5. It has a packaging bug. Please execute the following (required for Makefile, optional for CMake):
```
sudo apt-get install libturbojpeg
sudo ln -s /usr/lib/x86_64-linux-gnu/libturbojpeg.so.0.1.0 /usr/lib/x86_64-linux-gnu/libturbojpeg.so
```
