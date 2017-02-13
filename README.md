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

## 1. Camera calibration 
From the image below I find 9 coners in the x axis and 6 coners in the y axis. I scaled down the images from 1280 to 720:
![No of Corners](https://github.com/parthasen/SDC/blob/P4/output_images/0.png)

Based on [x=9,y=6] points calibration is done as in cell 5,I assumed z=0 of [x,y,z] plane. *objp* is used prepare object points. *objpoints* is used for 3d points in real world space and *imgpoints* for 2d points in image plane. *cv2.findChessboardCorners(gray, (9,6), None)* is used to  find the chessboard corners. *cv2.drawChessboardCorners(img, (9,6), corners, ret)* is used to draw and display the corners

![Calibration result](https://github.com/parthasen/SDC/blob/P4/output_images/1.png)

## 1. Remove distortion from images
**cv2.undistort(img, mtx, dist, None, mtx)** calculates camera calibration matrix and distortion coefficients to remove distortion from an image and output the undistorted image. The images used here are in **test_images** for testing pipeline on single frame. **objpoints, imgpoints** are used here again in **cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)**.

![Calibration result](https://github.com/parthasen/SDC/blob/P4/output_images/3.png)

## 3.  Color And Gradiant Threshold
Three functions used those apply Sobel x or y, then takes an absolute value and applies a threshold. I have applied following steps in these function: 
1)  Convert to grayscale
2)  Take the derivative in x or y given orient = 'x' or 'y'
3)  Take the absolute value of the derivative or gradient
4)  Scale to 8-bit (0 - 255) then convert to type = np.uint8
5)  Create a mask of 1's where the scaled gradient magnitude 
            is > thresh_min and < thresh_max
6)  Return this mask as your binary_output image
** abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(30, 130)) ** function used to get  absolute value and applies a threshold.
![Calibration result](https://github.com/parthasen/SDC/blob/P4/output_images/4.png)
![Calibration result](https://github.com/parthasen/SDC/blob/P4/output_images/5.png)

** mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)) ** function used to compute the magnitude of the gradient and applies a threshold.
![Calibration result](https://github.com/parthasen/SDC/blob/P4/output_images/6.png)

**dir_threshold(img, sobel_kernel=15, thresh=(0, np.pi/2))** function used to compute the direction of the gradient and applies a threshold.
![Calibration result](https://github.com/parthasen/SDC/blob/P4/output_images/7.png)

All those absolute value, magnitude, direction  and thresholds the S-channel of HLS are used for combined output.
![Calibration result](https://github.com/parthasen/SDC/blob/P4/output_images/9.png)

Comparing all those tests I found combind output using multiple thresholds can be used further.

## 4.  Apply a perspective transform
'''
    src = np.float32([[308,260],[408,260],[50,380],[650,380]])
    dst = np.float32([[0,0],[720,0],[0,405],[720,405]])
'''
First image was undistorted, <del>cv2.getPerspectiveTransform(src, dst)</del>
