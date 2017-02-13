The goals / steps of this project are the following:

1.  camera calibration 
2.  Remove distortion from images
3.  Color And Gradiant Threshold: use of color transforms, gradients, etc., to create a thresholded binary image.
4.  Apply a perspective transform to rectify binary image ("birds-eye view").
5.  Detect lane lines
6.  Determine the lane curvature
7.  Warp the detected lane boundaries back onto the original image.
8.  Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

# Output Video: 
[![Video Output](https://i.ytimg.com/vi/_u6I9w6048w/3.jpg?time=1486986474285)](https://www.youtube.com/watch?v=_u6I9w6048w)

# 1. Camera calibration 
From the image below I find 9 coners in the x axis and 6 coners in the y axis. I scaled down the images from 1280 to 720:
![No of Corners](https://github.com/parthasen/SDC/blob/P4/output_images/0.png)


![Calibration result](https://github.com/parthasen/SDC/blob/P4/output_images/1.png)

