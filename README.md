# project
It's C++ implementation of Accurate Depth Map Estimation from a Lanslet Light Field Camera from the following paper:

Jeon, Hae-Gon, et al. "Accurate depth map estimation from a lenslet light field camera." 
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.

Original MATLAB code can be found here: ( https://sites.google.com/site/hgjeoncv/home/depthfromlf_cvpr15 )

Note:
This package also includes part of following softwares:
- gco-v3.0: Multi-label optimization (http://vision.csd.uwo.ca/code/)
- Fast cost volume filtering: (https://www.ims.tuwien.ac.at/publications/tuw-210567)
- Fast weighted median filter: (http://www.cse.cuhk.edu.hk/~leojia/projects/fastwmedian/index.htm)
- Domain transform for edge-aware image and video processing: ( http://inf.ufrgs.br/~eslgastal/DomainTransform/ )

Required Libraries:
- Armadillo: C++ linear algebra library  (http://arma.sourceforge.net)
- Open CV 3.0: ( http://opencv.org )
