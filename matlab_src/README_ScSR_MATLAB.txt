MATLAB ScSR 

Demo scripts:

(0a)	Demo_DictionaryTraining.m
(0b) 	Demo_ScSR.m
(1) 	Demo_Dictionary_Comparison.m
(2) 	Demo_ThirdOrderFeatures.m


Details:

(0a) Demo_DictionaryTraining.m
 - runs coupled dictionary training using specified training images and dictionary parameters


(0b) Demo_ScSR.m
 - runs ScSR single-image super-resolution using learned dictionary


(1) Demo_Dictionary_Comparison.m

 - performs dictionary training using different parameters (# of atoms = 256, 512, 1024; # of randomly sampled image patches = 25k, 50k, 100k)
 - performs ScSR super-resolution reconstruction/restoration of test image from Yang et al. (2010)
 - Computes image quality metrics for ScSR recons using different dictionaries


(2) Demo_ThirdOrderFeatures.m

 - performs dictionary training using additional third-order features of LR image patches
 - performs ScSR super-resolution reconstruction/restoration of test image from Yang et al. (2010) using (a.) first-, second- and third-order LR image patch features and (b.) only first- and second-order LR image patch features
 - compares ScSR results using first- and second-order features to results using first-, second- and third-order features




References:

Jianchao Yang et al. “Image Super-Resolution Via Sparse Representation”. In: IEEE Trans- actions on Image Processing 19.11 (Nov. 2010), pp. 2861–2873. ISSN: 1057-7149. DOI: 10. 1109/TIP.2010.2050625. URL: http://ieeexplore.ieee.org/document/5466111/ (visited on 02/23/2024). 