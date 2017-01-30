# P3: Behavioural Cloning

#### The objective is to train a model that teaches a car to drive around a track in Udacity's simulator.
#### A common problem in Deep Learning is overfitting on a given training set. Overfitting by simply memorizing the images and their corresponding steering angles is common and to combat this two tracks on which to train and test. Target of this project is to drive the car successfully on both tracks.

#### Index: 
1.Files in this repo

2.Dataset

3.Preprocessing and Approach

4.Model architecture

#### 1.Files in this repo

model.py: Python script to read data, train model and save model.

model.json: Model architecture.

model.h5: Model weights (Large file, > 300MB).

drive.py: Python script that tells the car in the simulator how to drive

data/data: file with training data,attributes such as 'steering angle' mapped to image paths in driving_log.csv.Images in IMG/.
        
preprocess.ipynb: Note book for data visualization

model.ipynb: Notebook used to develop the model.

#### Dataset
Sample dataset downloaded from  https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip Udacity resource link is used for this project.
