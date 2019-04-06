## 1. 查看cuda版本
#### 1.1 访问文件
    >>> cat  /usr/local/cuda/version.txt
    CUDA Version 8.0.61
#### 1.2 使用nvcc
    >>> nvcc --version
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2016 NVIDIA Corporation
    Built on Tue_Jan_10_13:22:03_CST_2017
    Cuda compilation tools, release 8.0, V8.0.61

## 2. 查看cuDNN版本
    >>> cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
    #define CUDNN_MAJOR      6
    #define CUDNN_MINOR      0
    #define CUDNN_PATCHLEVEL 21
    --
    #define CUDNN_VERSION    (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

    #include "driver_types.h"
> 由上面可见，进行简单的加减法，像我的版本就是6.0.21