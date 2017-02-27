**SDC:Vehicle Detection along with Lane Line finding**

This model detects lane and cars both from a video stream. After detection model will draw blue rectangles around the cars detected by the model and green lane. 
 
**The steps of this project are the following:**

* 1.Reading Data 
* 2.Exploration
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images, color transformation and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Train a classifier Linear SVM classifier
* Normalization of features and randomize a selection for training and testing.
* Implement a sliding-window technique
* Use of trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

**All the code is implemented in the Jupyter Notebook `P5.ipynb` and output of the code is in `output_images` folder**

https://github.com/parthasen/SDC/blob/P5/P5-6.ipynb

https://github.com/parthasen/SDC/tree/P5/output_images


**Here I addressed each point in my implementation like:**  

1. I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

            cars = glob.glob("./vehicles/*/*/*.png")
            notcars = glob.glob("./non-vehicles/*/*/*.png")
 
Randomly selecting and ploting same from `vehicle` class:

 ![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/1.png)
 
Randomly selecting and ploting same from `non-vehicle` class:

 ![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/2.png) 
 
 ![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/3.png) 
 
 2. I have Computed the histogram of the RGB channels separately.
 
        rhist = np.histogram(img[:,:,0], bins=32, range=(0, 256))
        ghist = np.histogram(img[:,:,1], bins=32, range=(0, 256))
        bhist = np.histogram(img[:,:,2], bins=32, range=(0, 256))
        # Generating bin centers
        bin_edges = rhist[1]
        bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))

![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/4.png)
  
I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.
  # Define HOG parameters
      orient = 9
      pix_per_cell = 8
      cell_per_block = 2
      
      import skimage
      if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image      
