**SDC:Vehicle Detection along with Lane Line finding**

This model detects lane and cars both from a video stream. Model will draw blue rectangles around the cars detected by the model and green lane. 

[![Video Output](https://i.ytimg.com/vi/cuYiJWNH3WE/2.jpg?time=1488204419880)](https://www.youtube.com/watch?v=cuYiJWNH3WE&feature=youtu.be)


[![Video Output](https://i.ytimg.com/vi/DR7rsjxe2Ng/2.jpg?time=1488203947555)](https://www.youtube.com/watch?v=DR7rsjxe2Ng)


Histogram of Oriented Gradients (HOG) is used for feature extraction and trained Linear SVM classifier is used after normalization of features and randomizied selection.Then I have implemented a sliding-window technique and trained SVM classifier to search for vehicles in images. Then I run video on my pipeline on a video stream after using threshold of 5 and creating a heat map of recurring detections frame by frame to reject outliers.

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

https://github.com/parthasen/SDC/blob/P5/P5.ipynb

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
The sliding windows are created using different scales. Sliding windows are perspective, near the horizon are smaller, sliding windows closer to the camera car are larger. A total of four different scales have been used. Lastly the sliding wondows were tested with `xy_window=(192, 192)`, `xy_overlap=(0.75, 0.75))`. Overlap was adjested from 0.5.

        windows = slide_window(car_image, x_start_stop=[None, None], y_start_stop=[400, 500], 
                    xy_window=(64,64), xy_overlap=(0.75, 0.75))

        windows += slide_window(car_image, x_start_stop=[None, None], y_start_stop=[400, 500], 
                    xy_window=(100, 100), xy_overlap=(0.75, 0.75))
        windows += slide_window(car_image, x_start_stop=[None, None], y_start_stop=[400, 500], 
                    xy_window=(140,140), xy_overlap=(0.75, 0.75))
        windows += slide_window(car_image, x_start_stop=[None, None], y_start_stop=[430, 550], 
                    xy_window=(192, 192), xy_overlap=(0.75, 0.75))
        windows += slide_window(car_image, x_start_stop=[None, None], y_start_stop=[460, 580], 

I used `'RGB2YCrCb` and 3 individual channels HOG features for the entire image 
  
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

and then these two functions  `svc.predict()` `cv2.rectangle()` helped to find cars.

I found few **false positives** which can be removed using threshold to get heatmap. Essentially any detections that are not (partially) covered by a minimum number of sliding windows is discarded. The heatmap also helps combine duplicate detections into a single detection. For this project I used a heatmap threshold of 5.

![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/9.png) 


![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/10.png)

##### 7.  Finding Lane Lines
I used https://github.com/parthasen/SDC/blob/P4/P4.ipynb code to get pipeline. 

##### 8.  Combined pipeline on a video stream. 
Finaly `from scipy.ndimage.measurements import label` is used to determine the number of vehicles and, more importantly, their bounding boxes. 

![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/12.png)
![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/13.png)
![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/14.png)
![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/11.png)
![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/16.png)
![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/17.png)


Same pipeline  `process_video(image)`is applied to videos for detecting vehicles. ( notebook 26).

https://github.com/parthasen/SDC/blob/P5/output_images/project_video_output.mp4

I have modified the pipeline to detect both **vehicles and lane** using `pipeline(image)` (notebook 31).

https://github.com/parthasen/SDC/blob/P5/output_images/project_video_output_comb.mp4

Modified pipeline is done like `img` an output of detected vehicle image as input to lane line finding pipeline after resizing `cv2.resize(img)`

**Output Video:** 

[![Video Output](https://i.ytimg.com/vi/cuYiJWNH3WE/2.jpg?time=1488204419880)](https://www.youtube.com/watch?v=cuYiJWNH3WE&feature=youtu.be)


[![Video Output](https://i.ytimg.com/vi/DR7rsjxe2Ng/2.jpg?time=1488203947555)](https://www.youtube.com/watch?v=DR7rsjxe2Ng)



### Discussion and Challenges

Most of the functions are from lectures. Clubbing those together helped me to create a working pipeline. Briefly,I did a Histogram of Oriented Gradients (HOG) feature extraction and trained Linear SVM classifier after normalization of                     features and randomizied selection.Then I have implemented a sliding-window technique and trained SVM classifier to search for vehicles in images. Then I run video on my pipeline on a video stream after using threshold of 5 and creating a heat map of recurring detections frame by frame to reject outliers.

**Challenges**

1. Parameter tuning and finding the right combinations.
 Tuning was conducted on all test images. After achieving good results on the images, the pipeline was applied to the video. 
2. Video processing was time taking. 
3. Sliding windows grid and heatmap threshold were another parameters to optimize. Previously tuned, optimized, parameters were disturbed due to these. And another change was needed.
4. The result contains few false positives
5. The rectangles to identify vehicles is bit jittery.
6. Unable to classify multiple non car objects such as pedestrians from cars.

*In future I'll try to make more robust by:*

1. Better error handling and application of class.
2. Additional feature extraction using a grid search approach, removing duplicate and highly correlated features, etc.
3. Application of Trees and CNN for better classification.
4. Better data augmentation (e.g rotating, flipping images) to prevent false positives. 
5. Improvement of pipeline to work on own video. 
6. To Use voxelization techiques to speed up the vehicle detection model and rendering of the 3D model of the scene.
https://developer.nvidia.com/content/basics-gpu-voxelization


