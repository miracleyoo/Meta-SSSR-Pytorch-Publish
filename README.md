Official implementation of **Hyperspectral Image Super-Resolution in Arbitrary Input-Output Band Settings(IEEE/CVF Winter Conference on Applications of Computer Vision(WACV) 2022)(PyTorch)**

Meta-SSSR means META-learning-based Spacial-Spectral Super-Resolution.

If you have similar problem regarding dealing with data whoes input/output has unfixed/random/arbitrary number of channels/bands, this repository can possibly help.

Our code is built on [Meta-SR](https://github.com/XuecaiHu/Meta-SR-Pytorch) which is based on [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

# Requirements

* Pytorch 0.4.0
* Python 3.5
* numpy
* skimage
* imageio
* cv2  

*note that if you use another version of pytorch (>0.4.0), you can rewrite the dataloader.py

# Update notes
- TODO


# Install and run demo
1. download the code
```
git clone https://github.com/miracleyoo/Meta-SSSR-Pytorch-Publish.git
cd Meta-SSSR-Pytorch-Publish
```


2. Structure:

   1. `main.py`: The main entrance of the code.
   2. `trainer.py`: Responsible for the train part.
   3. `test.py`: The entrance of testing.


# Citation

Please make sure to cite our paper if you use our code.

```
@inproceedings{zhang2022hyperspectral,
  title={Hyperspectral Image Super-Resolution in Arbitrary Input-Output Band Settings},
  author={Zhang, Zhongyang and Xu, Zhiyang and Ahmed, Zia and Salekin, Asif and Rahman, Tauhidur},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={749--759},
  year={2022}
}
```
# Contact
Zhongyang Zhang(zhz138@ucsd.edu)
