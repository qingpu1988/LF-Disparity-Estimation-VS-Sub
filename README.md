# LF-Disparity-Estimation-VS-Sub
LF disparity estimation dataset and view-selective convolution-based cost volume construction method
We propose the TO-LF dataset, designed to address challenges in complex occlusion scenarios and textureless region handling, which provides high-quality RGB light field images paired with accurate disparity maps. The TO-LF dataset comprises 60 training images, 6 validation images, and 12 test images.  The test set is categorized into "Texture" and "Occlusion" subsets. The "Texture" subset involves scenes with textureless areas or repetitive patterns, whereas the "Occlusion" subset contains scenes with complex self-occlusions or depth discontinuities， as illustrated below.
![Representative samples from the proposed TO-LF dataset.](https://github.com/qingpu1988/LF-Disparity-Estimation-VS-Sub/blob/main/fig1.png) 
The train set and validation set can be downloaded in [Baidu Drive](https://pan.baidu.com/s/14pvZdMePc57S2UBqgAjxZg?pwd=dhuu), [OneDrive](https://1drv.ms/f/c/e4281d7ece40bab0/Ev0yPjo4_eJGpBlSJB-Vip4BLpTBY8hcmZWK8pm9j-ehxw?e=b0lSUb). If you are not able to download from these two websites, please contact: zhoushubo@dhu.edu.cn.

The GT disparity maps for test scenes are AES-encoded, and a Matlab evaluation script  (.p file) is provided for standardized evaluation. ****The test set and evaluation algorithm will be released soon after the manuscript is accpted****

 We also propose VS-Sub for LF disparity estimation, which upsamples LF feature in the macropixel image (MacPI) domain and employs a viewpoint selection layer (VS Conv) to construct the subpixel cost volume, as illustarated below.

 ![ The VS-Sub model pipeline.](https://github.com/qingpu1988/LF-Disparity-Estimation-VS-Sub/blob/main/fig2.png)
 
****The code and pretrain model can be found in folder "VS-Sub", and the training code will be released soon after the manuscript is accpted**** This code is heavily borrowed from [DistgDisp](https://github.com/YingqianWang/DistgDisp). Many Thanks! 

we also retraied several learning-based methods on our TO-LF dataset. The pretarined models can be found in folder "Pretrained".
