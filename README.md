**SDC:Vehicle Detection along with Lane Line finding**

            Thid model detects lane and cars both from a video stream. After detection model will draw blue rectangles around the cars detected by the model and green lane. 
 
*The steps of this project are the following:*

* 1.Reading Data 
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images, color transformation and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Train a classifier Linear SVM classifier
* Normalization of features and randomize a selection for training and testing.
* Implement a sliding-window technique
* Use of trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

1. I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

            cars = glob.glob("./vehicles/*/*/*.png")
            notcars = glob.glob("./non-vehicles/*/*/*.png")
 
Randomly selecting and ploting same from car class:

 ![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/1.png)
 
Randomly selecting and ploting same from not car class:

 ![Calibration result](https://github.com/parthasen/SDC/blob/P5/output_images/2.png) 
