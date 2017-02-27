**SDC:Vehicle Detection along with Lane Line finding**

This model detects lane and cars both from a video stream. After detection model will draw blue rectangles around the cars detected by the model and green lane. 
 
**The steps of this project are the following:**

* 1.Reading Data 
* 2.Exploration
* 3.Perform a Histogram of Oriented Gradients (HOG) feature extraction 
* 4.Randomization,Normalization,Suffling of training and test set features.
* 5.Train a classifier Linear SVM classifier
* 6.Implement a sliding-window technique and use of trained classifier to search for vehicles in images.
* 7.Finding Lane Lines
* 8.Combined pipeline on a video stream 

**All the code is implemented in the Jupyter Notebook `P5.ipynb` and output of the code is in `output_images` folder**

https://github.com/parthasen/SDC/blob/P5/P5-6.ipynb

https://github.com/parthasen/SDC/tree/P5/output_images


**Here I addressed each point in my implementation like:**  

1. I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

            cars = glob.glob("./vehicles/*/*/*.png")
            notcars = glob.glob("./non-vehicles/*/*/*.png")
 
2. Randomly selecting and ploting same from `vehicle` class:

 ![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/1.png)
 
Randomly selecting and ploting same from `non-vehicle` class:

 ![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/2.png) 
 
 ![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/3.png) 
 
 3. I have Computed the histogram of the RGB channels separately.
 
        rhist = np.histogram(img[:,:,0], bins=32, range=(0, 256))
        ghist = np.histogram(img[:,:,1], bins=32, range=(0, 256))
        bhist = np.histogram(img[:,:,2], bins=32, range=(0, 256))
        # Generating bin centers
        bin_edges = rhist[1]
        bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))

![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/4.png)

*I have then plotted features*

      color1 = cv2.resize(img[:,:,0], size).ravel()
      color2 = cv2.resize(img[:,:,1], size).ravel()
      color3 = cv2.resize(img[:,:,2], size).ravel()
 
![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/6.png) 

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.
*I have settled to these HOG parameters after playing with the parameters colorspace, orient,pix_per_cell, cell_per_block, and hog_channel to get a feel for what combination of parameters give the best results.Tweak these parameters and see how the results change*.
        
    color_space= 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    spatial_size = (16, 16)
    hist_bins = 32
    hist_range = (0, 256)
    orient = 8
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
      
      import skimage
      if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
  
![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/5.png)

I found the HOG features lastly.
   
    car_features = extract_features(cars, cspace=color_space, spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range,
                    orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    notcar_features = extract_features(notcars, cspace=color_space, spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range,
                    orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)

4. Randomization,Normalization,Suffling of training and test set features
    
First I normalized `car_features` and `notcar_features`.
      
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    
![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/7.png)

Then I have created training and test set randomly (test set 20%)

    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=12412)
    X_train, y_train = shuffle(X_train, y_train, random_state=2342)
5. Linear SVM classification.      
 
       Lastly I tried SVM classification. First I tried Linear SVM classification with spatial and histogram only `Accuracy of SVC based on spatial and histogram only=  0.9155` but that accuracy was low so I tried later with HOG features and `Test Accuracy of HOG based SVC =  0.9885` was acceptable.

I tested the prediction.

    Using spatial binning of: (16, 16) and 32 histogram bins
    Feature vector length: 5568
    My SVC predicts:  [ 0.  1.  0.  0.  0.  0.  1.  1.  1.  1.]
    For these 10 labels:  [ 0.  1.  0.  0.  0.  0.  1.  1.  1.  1.]
    0.0163 Seconds to predict 10 labels with SVC
![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/8.png) 
       
6. Implement a sliding-window technique
I used `'RGB2YCrCb` and 3 individual channels HOG features for the entire image 
  
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
and then these two functions  `svc.predict()` `cv2.rectangle()` helped to find cars.

I found few false positives which can be removed using threshold to get heatmap:

![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/9.png) 

Lastly the sliding wondows were tested with `xy_window=(192, 192)`, `xy_overlap=(0.75, 0.75))`. Overlap was adjested from 0.5.


![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/10.png)

7.  Finding Lane Lines
I used https://github.com/parthasen/SDC/blob/P4/P4.ipynb code to get pipeline. 

8.  Combined pipeline on a video stream. 

 
        ksize = 3
        def pipeline(image):
        global heat, heat_list
    car_image = image.astype(np.float32)/255
    heatmap_factor = 0.9

    windows = slide_window(car_image, x_start_stop=[None, None], y_start_stop=[400, 500], 
                    xy_window=(64,64), xy_overlap=(0.75, 0.75))

    windows += slide_window(car_image, x_start_stop=[None, None], y_start_stop=[400, 500], 
                    xy_window=(100, 100), xy_overlap=(0.75, 0.75))
    windows += slide_window(car_image, x_start_stop=[None, None], y_start_stop=[400, 500], 
                    xy_window=(140,140), xy_overlap=(0.75, 0.75))
    windows += slide_window(car_image, x_start_stop=[None, None], y_start_stop=[430, 550], 
                    xy_window=(192, 192), xy_overlap=(0.75, 0.75))
    windows += slide_window(car_image, x_start_stop=[None, None], y_start_stop=[460, 580], 
    
    xy_window=(192, 192), xy_overlap=(0.75, 0.75))
    
    hot_windows = search_windows(car_image, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel)
    heat_list.append(hot_windows)
    # Create heatmap
    heat_image = np.zeros_like(car_image)
    
    if len(heat_list) > 15:
        heat_list.pop(0)
        for hot_window in heat_list:
            for window in hot_windows:
                heat_image[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 5
    elif len(heat_list) < 15:
        for window in hot_windows:
            heat_image[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 5
    
    heat_image = cv2.GaussianBlur(heat_image,(5,5),0)
    
    heat_image = apply_threshold(heat_image, 5) #4
    
    if heat == None:
        heat = heat_image
    else:
        #heat_image = heat * heat_factor + heatmap_image * (1 - heatmap_factor)
        heat_image = apply_threshold(heat_image, 10) #6
        heat = heat_image
 
    labels = label(heat_image)
    img = draw_labeled_bboxes(image, labels)
    
    img = cv2.resize(img, (720, 405))
    apex, apey = 360, 258
    offset = 0
    offset_far = 50
    offset_near = 10
    src = np.float32([[int(apex-offset_far),apey],
                  [int(apex+offset_far),apey],
                  [int(0+offset_near),390],
                  [int(720-offset_near),390]])
    dst = np.float32([[0,0],[720,0],[0,405],[720,405]])
    #img = cv2.imread('test_images/straight_lines1.jpg')
    img_size = (img.shape[1],img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    img = cv2.undistort(img, mtx, dist, None, mtx)
    img= cv2.GaussianBlur(img, (3,3), 0)
    M = cv2.getPerspectiveTransform(src, dst)
    Mi = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gradx = abs_sobel_thresh(warped, orient='x', sobel_kernel=3, thresh=(10, 230))
    grady = abs_sobel_thresh(warped, orient='y', sobel_kernel=3, thresh=(10, 230))
    mag_binary = mag_thresh(warped, sobel_kernel=ksize, mag_thresh=(30, 150))
    dir_binary = dir_threshold(warped, sobel_kernel=ksize, thresh=(0.7, 1.3))
    hls_binary = HLS_single(warped,thresh=(90, 255))
    image = np.zeros_like(dir_binary)
    image[((gradx == 1) & (hls_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
          
    leftx, lefty, rightx, righty, out_img = left_right_lane(image)
        
    yvals = np.linspace(0, img.shape[0], num=img.shape[0])

    # Fit a second order polynomial to each lane line
    left_fit = np.polyfit(lefty, leftx, 2)
    left_fitx = left_fit[0]*yvals**2 + left_fit[1]*yvals + left_fit[2]
    right_fit = np.polyfit(righty, rightx, 2)
    right_fitx = right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/405 # meters per pixel in y dimension
    xm_per_pix = 3.7/600 # meteres per pixel in x dimension
    # Define y-value where we want radius of curvature
    # I'll choose 3 y-values(max, mean and min):
    y_eval = np.max(yvals)
    
    y_eval1 = np.max(yvals)
    y_eval2 = np.mean(yvals)
    y_eval3 = np.min(yvals)
    left_fitx_1 = left_fit[0]*y_eval1**2 + left_fit[1]*yvals + left_fit[2]
    left_fitx_2 = left_fit[0]*y_eval2**2 + left_fit[1]*yvals + left_fit[2]
    left_fitx_3 = left_fit[0]*y_eval3**2 + left_fit[1]*yvals + left_fit[2]
    right_fitx_1 = right_fit[0]*y_eval1**2 + right_fit[1]*yvals + right_fit[2]
    right_fitx_2 = right_fit[0]*y_eval2**2 + right_fit[1]*yvals + right_fit[2]
    right_fitx_3 = right_fit[0]*y_eval3**2 + right_fit[1]*yvals + right_fit[2]
    
    
    # Calculated the turning center point xc, yc and radius: 
            
    lm1, lm2, lxc, lyc, lradius = c_radius(left_fitx_1,y_eval1,left_fitx_2,y_eval2,left_fitx_3,y_eval3,)
    l_steering_angle = 4*360/lxc # assume xc <> 0, xc and radius value is very close, xc will show the direction as well
    
    
    rm1, rm2, rxc, ryc, rradius = c_radius(right_fitx_1,y_eval1,right_fitx_2,y_eval2,right_fitx_3,y_eval3,)
     
    r_steering_angle = 4*360/rxc # assume xc <> 0, xc and radius value is very close, xc will show the direction as well
    steering_angle = l_steering_angle + r_steering_angle
    turning_radius = (lradius+rradius)/2 # smooth out the radius
    
    # Find camera position
    left_mean = np.mean(leftx)
    right_mean = np.mean(rightx)
    camera_pos = (image.shape[1]/2)-np.mean([left_mean, right_mean])

    left_fit_cr = np.polyfit(np.array(lefty,dtype=np.float32)*ym_per_pix, \
                         np.array(leftx,dtype=np.float32)*xm_per_pix, 2)
    right_fit_cr = np.polyfit(np.array(righty,dtype=np.float32)*ym_per_pix, \
                          np.array(rightx,dtype=np.float32)*xm_per_pix, 2)
    
    # Return radius of curvature is in meters
    
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) \
                             /np.absolute(2*left_fit_cr[0])

    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) \
                                /np.absolute(2*right_fit_cr[0])
      
    
    # Link all points for cv2.fillPoly() in pix space
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))
    #pts_left = np.array([np.transpose(np.vstack([left_arcx_pt, yvals]))])
    #pts_right = np.array([np.flipud(np.transpose(np.vstack([right_arcx_pt, yvals])))])
    #pts = np.hstack((pts_left, pts_right))
    # pts = np.array([pts], dtype=np.int32)
    
    # Draw the lane onto the warped blank image
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    cv2.fillPoly(warp_zero, np.int_([pts]), (0,255, 0))
    cv2.polylines(warp_zero, np.array([pts_left], dtype=np.int32), False,(255,0,0),thickness = 15)
    cv2.polylines(warp_zero, np.array([pts_right], dtype=np.int32), False,(0,0,255),thickness = 15)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    
    cv2.putText(img,'Camera Position::{:.2f}m'.format(camera_pos*xm_per_pix),(10,30), font, 1,(255,255,255),2)
    cv2.putText(img,'Turning Radius::{}m'.format(camera_pos*xm_per_pix),(10,60), font, 1,(255,255,255),2)
    cv2.putText(img,'Steering Angle:]{:.6}deg'.format(str(steering_angle)),(10,90), font, 1,(255,255,255),2)
   
    # Warp back to original view
    unwarp =cv2.warpPerspective(warp_zero, Mi,img_size)
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, unwarp, 0.3, 0)
    #result = cv2.resize(result, (720, 405))

    return result
