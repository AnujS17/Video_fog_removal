# Hybrid Learning DehazeNet for Multi Image Haze Removal 

**Wei-Ting Chen, Hao-Yu Feng, Jian-Jiun Ding, Sy-Yen Kuo**  
[[Paper Download]](https://ieeexplore.ieee.org/document/9094006)



# Abstract:

Images captured in a hazy environment usually suffer from bad visibility and missing information. Over many years, learning-based and handcrafted prior-based dehazing algorithms have been rigorously developed. However, both algorithms exhibit some weaknesses in terms of haze removal performance. Therefore, in this work, we have proposed the patch-map-based hybrid learning DehazeNet, which integrates these two strategies by using a hybrid learning technique involving the patch map and a bi-attentive generative adversarial network. In this method, the reasons limiting the performance of the dark channel prior (DCP) have been analyzed. A new feature called the patch map has been defined for selecting the patch size adaptively. Using this map, the limitations of the DCP (e.g., color distortion and failure to recover images involving white scenes) can be addressed efficiently. In addition, to further enhance the performance of the method for haze removal, a patch-map-based DCP has been embedded into the network, and this module has been trained with the atmospheric light generator, patch map selection module, and refined module simultaneously. A combination of traditional and learning-based methods can efficiently improve the haze removal performance of the network. Experimental results show that the proposed method can achieve better reconstruction results compared to other state-of-the-art haze removal algorithms.


```
