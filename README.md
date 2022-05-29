# Robust-Counterfactually-Fair-Classification
[CLeaR‘2022](https://www.cclear.cc/2022): Fair Classification with Instance-dependent Label Noise (PyTorch implementation).



This is the code for the paper:
[Fair Classification with Instance-dependent Label Noise](https://openreview.net/pdf?id=s-pcpETLpY)      
Songhua Wu, Mingming Gong, Bo Han, Yang Liu, Tongliang Liu.



If you find this code useful for your research, please cite  
```bash
@inproceedings{wu2021fair,
  title={Fair Classification with Instance-dependent Label Noise},
  author={Wu, Songhua and Gong, Mingming and Han, Bo and Liu, Yang and Liu, Tongliang},
  booktitle={First Conference on Causal Learning and Reasoning},
  year={2022}
}
```



## Dependencies
We implement our methods by PyTorch on Nvidia GeForce RTX 2080 Ti. The environment is as bellow:
- [Ubuntu 20.04 Desktop](https://ubuntu.com/download)
- [PyTorch](https://PyTorch.org/), version >= 0.4.1
- [CUDA](https://developer.nvidia.com/cuda-downloads), version >= 9.0
- [Anaconda3](https://www.anaconda.com/)



## Datasets

We process the raw data into *.npy* format, which can be found in the */dataset* folder of this repository.



## Runing the code
Here is an example for RCFC: 

```bash
python RCFC.py --dataset adult
```


Here is an example for R-*p*-Fair: 

```bash
python rpfair.py --dataset adult
```
