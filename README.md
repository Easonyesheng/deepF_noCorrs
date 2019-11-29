# DeepFMatrix
Try to use Deep Learning to obtain fundamental matrices.

# References:

## Stereo Image Estimation
[DeMoN(CVPR 2017)](https://arxiv.org/pdf/1612.02401.pdf)

## H-Matrix Estimation 
[Deep Image Homography Estimation](https://arxiv.org/abs/1606.03798)

[Cascaded Lucas-Kanade Networks(CVPR2017)](http://openaccess.thecvf.com/content_cvpr_2017/papers/Chang_CLKN_Cascaded_Lucas-Kanade_CVPR_2017_paper.pdf)

[Direct Regression of H-Matrix](https://arxiv.org/pdf/1709.03524.pdf)

## Camera Pose Estimation
[VINet](http://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14462/14272)

[Relative Camera Pose Estimation](https://arxiv.org/pdf/1702.01381.pdf)

[PosNet](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.pdf)


## Geometrics Matching
[CNN for Geo-Matching](https://arxiv.org/pdf/1703.05593.pdf)

# for training the model:
	run the file single_fnet.py
	the arguments to be specified are mentioned in the same file.
	The directionary structure:
	--deepF_noCorrs
		--code
			--all code thing & mkdir data here
		--data(the same data dir as in code)
		--saved_npy
		
# Code Parser  
How does this single_fnet.py work?  
Get data --> training & testing & evaluating --> save  
## Data    
	data_util.py  
	  use make_kitti_data_loader() to input left & right images   
	  use get_FMat() in kitti_fmat.py to get the F_GT.  
	  use data_spliter() to devide data to training & testing & evaluation.  
	  save to ../saved_npy/ as .npy file.  
## Test
	A dictionary named metrics in single_fnet.py, whose items are functions used to evaluate F matrix.  
	use a class named KPCorrBasedMetric() in evaluation.py to pack these functions make their output 3 values as they output 1 value before.  
	the evaluation functions are in ./povary/stereo_pairs.py.  
	the final outputs when test or evaluation are 1.|score of predicted - score of F_GT| 2.score of predicted F 3.score of F_GT  
	
