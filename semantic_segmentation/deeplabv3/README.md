Dataset:

We will use the PASCAL VOC 2012 dataset for the segmentation task: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/


Create folder 'data_voc' in project deeplabv3.

Upload dataset and copy following folders : ImageSets, JPEGImages,SegmentationClass in the data folder:

+-- depplabv3
|   +-- data_voc
        +-- ImageSets
        +-- JPEGImages
        +-- SegmentationClass
|   +-- src
|   



@misc{pascal-voc-2012,
	author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
	title = "The {PASCAL} {V}isual {O}bject {C}lasses {C}hallenge 2012 {(VOC2012)} {R}esults",
	howpublished = "http://www.pascal-network.org/challenges/VOC/voc2012/workshop/index.html"}