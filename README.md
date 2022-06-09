# MSC Net Reproduction

Workspace for 2022L WB Transfer Learning

*Authors: Andrzej Pióro, Mikołaj Spytek, Kinga Ułasik*

## Abstract
Multitask learning is a developing field of knowledge transfer used for solving multiple dependent tasks. It proves itself helpful in medical applications where various tasks are based on common features, and their combination may improve results. In the "MSC-Net: Multitask Learning Network for Retinal Vessel Segmentation and Centerline Extraction" paper, an architecture based on a cross-stitch network was presented and used for automatic segmentation and centerline extraction of blood vessels from retinal fundus images. We challenged ourselves to reproduce the architecture depicted in the mentioned paper and compare our results to those reported by the authors.

## Repository content

This repository contains all the scripts necessary for reproducing our work. Additionally, the networks' most optimal weights are included in the *weights* directory if one wants to run the ready model without needing to train it. 
Preprocessed data from publicly available, mainstream  STARE, DRIVE, and 	CHASE datasets are located in the *data* directory. 


### Source article
Pan, L., Zhang, Z., Zheng, S., & Huang, L. (2022). MSC-Net: Multitask Learning Network for Retinal Vessel Segmentation and Centerline Extraction. Applied Sciences (Switzerland): https://doi.org/10.3390/app12010403

