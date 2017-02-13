The goals / steps of this project are the following:

1.  camera calibration 
2.  Remove distortion from images
3.  Color And Gradiant Threshold: use of color transforms, gradients, etc., to create a thresholded binary image.
4.  Apply a perspective transform to rectify binary image ("birds-eye view").
5.  Detect lane lines
6.  Determine the lane curvature  and numerical estimation vehicle position.
7.  Warp the detected lane boundaries back onto the original image.
8.  Output visual display of the lane boundaries

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
**abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(30, 130))** function used to get  absolute value and applies a threshold.
![Calibration result](https://github.com/parthasen/SDC/blob/P4/output_images/4.png)
![Calibration result](https://github.com/parthasen/SDC/blob/P4/output_images/5.png)

**mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255))** function used to compute the magnitude of the gradient and applies a threshold.
![Calibration result](https://github.com/parthasen/SDC/blob/P4/output_images/6.png)

**dir_threshold(img, sobel_kernel=15, thresh=(0, np.pi/2))** function used to compute the direction of the gradient and applies a threshold.
![Calibration result](https://github.com/parthasen/SDC/blob/P4/output_images/7.png)

All those absolute value, magnitude, direction  and thresholds the S-channel of HLS are used for combined output.
![Calibration result](https://github.com/parthasen/SDC/blob/P4/output_images/9.png)

Comparing all those tests I found combind output using multiple thresholds can be used further.

## 4.  Apply a perspective transform

            src = np.float32([[308,260],[408,260],[50,380],[650,380]])
            dst = np.float32([[0,0],[720,0],[0,405],[720,405]])

are used to find the source and destination and used in **cv2.getPerspectiveTransform(src, dst)** for transformation. This was again used in *cv2.warpPerspective(undist, M, img_size)* to get transformed image. 

![Calibration result](https://github.com/parthasen/SDC/blob/P4/output_images/10.png)

            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(undist, M, img_size)
            ksize=3
            gradx = abs_sobel_thresh(warped, orient='x', sobel_kernel=ksize, thresh=(10, 230))
            grady = abs_sobel_thresh(warped, orient='y', sobel_kernel=ksize, thresh=(10, 230))
            mag_binary = mag_thresh(warped, sobel_kernel=ksize, mag_thresh=(30, 150))
            dir_binary = dir_threshold(warped, sobel_kernel=ksize, thresh=(0.7, 1.3))
            hls_binary = HLS_single(warped,thresh=(90, 255))
            combined = np.zeros_like(dir_binary)
            combined[((gradx == 1) & (hls_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
Above code is used to get warped image with combined gradient thresholds. 
![Calibration result](https://github.com/parthasen/SDC/blob/P4/output_images/11.png)

## 5.  Detect lane lines
1. Warped image after perspective transformation and after applying gradient thresholds is used to get the histogram.
2. 9 slideing windows are used to get x and y position 
3. 2nd order polynomial is calculated from the code below 

            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            
![Calibration result](https://github.com/parthasen/SDC/blob/P4/output_images/12.png)

![Calibration result](https://github.com/parthasen/SDC/blob/P4/output_images/13.png)

Using above 3 steps finally I find this:

![Calibration result](https://github.com/parthasen/SDC/blob/P4/output_images/14.png)

## 6.  Determine the lane curvature  and numerical estimation vehicle position.
            
I defined conversions in x and y from pixels space to meters:

     ym_per_pix = 30/405 # meters per pixel in y dimension
     xm_per_pix = 3.7/600 # meteres per pixel in x dimension
Then I defined y-value where we want radius of curvature.
            
     left_fit_cr = np.polyfit(np.array(lefty,dtype=np.float32)*ym_per_pix, \
                         np.array(leftx,dtype=np.float32)*xm_per_pix, 2)
     right_fit_cr = np.polyfit(np.array(righty,dtype=np.float32)*ym_per_pix, \
                          np.array(rightx,dtype=np.float32)*xm_per_pix, 2)
     left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) \
                             /np.absolute(2*left_fit_cr[0])
     right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) \
 
                                /np.absolute(2*right_fit_cr[0])
 Above codes are used get lane curvature:
 ![Calibration result](https://github.com/parthasen/SDC/blob/P4/output_images/15.png)
 
  I calculated the turning center point xc, yc and radius as below: 
            
      lm1, lm2, lxc, lyc, lradius = c_radius(left_fitx_1,y_eval1,left_fitx_2,y_eval2,left_fitx_3,y_eval3,)
      l_steering_angle = 4*360/lxc # assume xc <> 0, xc and radius value is very close, xc will show the direction as well
      rm1, rm2, rxc, ryc, rradius = c_radius(right_fitx_1,y_eval1,right_fitx_2,y_eval2,right_fitx_3,y_eval3,)
      r_steering_angle = 4*360/rxc # assume xc <> 0, xc and radius value is very close, xc will show the direction as well
      steering_angle = l_steering_angle + r_steering_angle
      turning_radius = (lradius+rradius)/2 # smooth out the radius
    
   And camera position:
   
       left_mean = np.mean(leftx)
       right_mean = np.mean(rightx)
       camera_pos = (combined.shape[1]/2)-np.mean([left_mean, right_mean])
 
 ## 7.  Warp the detected lane boundaries back onto the original image.
 Warp back to original view using Mi:
 
     M = cv2.getPerspectiveTransform(src, dst)
     Mi = cv2.getPerspectiveTransform(dst, src)
     unwarp =cv2.warpPerspective(warp_zero, Mi,img_size)
 Combine the result with the original image:
            
     result = cv2.addWeighted(img, 1, unwarp, 0.3, 0)
## 8.  Output visual display of the lane boundaries     
![Calibration result](https://github.com/parthasen/SDC/blob/P4/output_images/16.png)

https://github.com/parthasen/SDC/blob/P4/project_video_output.mp4

https://github.com/parthasen/SDC/blob/P4/challenge_video_output.mp4

https://github.com/parthasen/SDC/blob/P4/harder_challenge_video_out1.mp4

[![Video Output](https://i.ytimg.com/vi/_u6I9w6048w/3.jpg?time=1486986474285)](https://www.youtube.com/watch?v=_u6I9w6048w)

## Discussion
Where will your pipeline likely fail?
1. Line detection is not working well for harder and challenger videos. 
2. When light or shade is excessive than test images then pipeline is not working.
Briefly discuss any problems / issues you faced in your implementation of this project.
3. I used Udacity sliding window detector but other methods may be helpful. we can use machine learning approach instead. 
4. Speed is slightly increased by reducing image size. Speed was slowed down at first when I was using full size images. 
5. Tuning thresholds and finding shape of source and destination were time taking. 
6. Line fitting algorithm works worse on the dashed line.It's because there was not enough points to fit the line.
What could you do to make it more robust?
7. Smoothin using moving average
8. Working on qwn video after completion of term 1. 
9. More application of CNN 

