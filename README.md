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

##### 1. I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

            cars = glob.glob("./vehicles/*/*/*.png")
            notcars = glob.glob("./non-vehicles/*/*/*.png")
 
##### 2. Randomly selecting and ploting same from `vehicle` class:

 ![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/1.png)
 
Randomly selecting and ploting same from `non-vehicle` class:

 ![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/2.png) 
 
 ![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/3.png) 
 
 ##### 3.I have Computed the histogram of the RGB channels separately.
 
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

##### 4.Randomization,Normalization,Suffling of training and test set features
    
First I normalized `car_features` and `notcar_features`.
      
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    
![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/7.png)

Then I have created training and test set randomly (test set 20%)

    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=12412)
    X_train, y_train = shuffle(X_train, y_train, random_state=2342)
##### 5.Linear SVM classification.      
 
       Lastly I tried SVM classification. First I tried Linear SVM classification with spatial and histogram only `Accuracy of SVC based on spatial and histogram only=  0.9155` but that accuracy was low so I tried later with HOG features and `Test Accuracy of HOG based SVC =  0.9885` was acceptable.

I tested the prediction.

    Using spatial binning of: (16, 16) and 32 histogram bins
    Feature vector length: 5568
    My SVC predicts:  [ 0.  1.  0.  0.  0.  0.  1.  1.  1.  1.]
    For these 10 labels:  [ 0.  1.  0.  0.  0.  0.  1.  1.  1.  1.]
    0.0163 Seconds to predict 10 labels with SVC
![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/8.png) 
       
##### 6. Implement a sliding-window technique
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

##### 7.  Finding Lane Lines
I used https://github.com/parthasen/SDC/blob/P4/P4.ipynb code to get pipeline. 

###### 8.  Combined pipeline on a video stream. 

 
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
        result = cv2.addWeighted(img, 1, unwarp, 0.3, 0)
        
        ....

        return result
