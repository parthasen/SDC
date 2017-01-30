# P3: Behavioural Cloning

#### The objective is to train a model that teaches a car to drive around a track in Udacity's simulator.
#### A common problem in Deep Learning is overfitting on a given training set. Overfitting by simply memorizing the images and their corresponding steering angles is common and to combat this two tracks on which to train and test. Target of this project is to drive the car successfully on both tracks.

There are some common techniques you here to prevent overfitting:
1.Adding dropout layers to your network.

2.Splitting your dataset into a training set and a validation set.

 1. fit_generator function provided by Keras is used https://keras.io/models/model/
 2. dataset downloaded from  https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip
 3. 50Hz simulator is used https://files.slack.com/files-pri/T2HQV035L-F3A47CT7Z/download/simulator-linux.zip
 4. These codes helped me a lot 
 https://github.com/commaai/research/blob/master/train_steering_model.py
 https://github.com/udacity/self-driving-car/tree/master/steering-models
 5. These are my other helpful references 
  https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
  http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
  https://keras.io/preprocessing/image/
  https://keras.io/getting-started/functional-api-guide/
  https://github.com/commaai/research
  https://github.com/udacity/self-driving-car/blob/master/steering-models/community-models/chauffeur/models.py

#### Index: 
1.Files in this repo

2.Dataset

3.Preprocessing and Approach

4.Model architecture

#### 1.Files in this repo

model.py - The script used to create and train the model.
    drive.py - The script to drive the car. You can feel free to resubmit the original drive.py or make modifications 
    and submit your modified version.
    model.json - The model architecture.
    model.h5 - The model weights.
    README.md - explains the structure of your network and training approach.
    
model.py: Python script to read data, train model and save model.

model.json: Model architecture.

model.h5: Model weights (Large file, > 300MB).

drive.py: Python script that tells the car in the simulator how to drive

data/data: file with training data,attributes such as 'steering angle' mapped to image paths in driving_log.csv.Images in IMG/.
        
preprocess.ipynb: Note book for data visualization

model.ipynb: Notebook used to develop the model.

#### Dataset
Sample dataset downloaded from  https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip Udacity resource link is used for this project.
