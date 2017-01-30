# P3: Behavioural Cloning

#### The objective is to train a model that teaches a car to drive around a track in Udacity's simulator.
##### A common problem in Deep Learning is overfitting on a given training set. Overfitting by simply memorizing the images and their corresponding steering angles is common and to combat this two tracks are taken on which to train and test this model. Target of this project is to drive the car successfully on both tracks.

##### There are some common techniques used here to prevent overfitting:

1.Adding dropout layers to your network.

model.add(Dropout(0.5))

2.Splitting your dataset into a training set and a validation set. (90%)

split = (int(len(shuffled_data) * 0.9) // BATCH_SIZE) * BATCH_SIZE

train_data = data[:split]

train_data = remove_low_steering(train_data)

val_data = data[split:]

new_val = (len(val_data) // BATCH_SIZE) * BATCH_SIZE

val_data = val_data[:new_val]

samples_per_epoch = len(train_data) - BATCH_SIZE

##### fit_generator is used for generate data for training:

Data generated batch-by-batch by this Python generator. The generator is run in parallel to the model, for efficiency.

Code used:

model.fit_generator(generate_data(train_data), samples_per_epoch=samples_per_epoch, nb_epoch=EPOCHS, validation_data=generate_data(val_data), nb_val_samples=len(val_data))

#### Index: 
1.Files in this repo

2.Dataset

3.Preprocessing and Approach

4.Model architecture

5.Result

#### 1.Files in this repo

model.py - The script used to create and train the model.
https://github.com/parthasen/SDC/blob/P3/model.py

drive.py - The script to drive the car. You can feel free to resubmit the original drive.py or make modifications 
    and submit your modified version.
https://github.com/parthasen/SDC/blob/P3/drive.py    

model.json - The model architecture.
https://github.com/parthasen/SDC/blob/P3/model.json

model.h5 - The model weights.
https://github.com/parthasen/SDC/blob/P3/model.h5

README.md - explains the structure of your network and training approach.

preprocess.ipynb: Note book for data visualization
https://github.com/parthasen/SDC/blob/P3/preprocess.ipynb

model.ipynb: Notebook used to develop the model.
https://github.com/parthasen/SDC/blob/P3/model.ipynb

#### 2.Dataset
Sample dataset downloaded from  https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip Udacity resource link is used for this project.
Dataset has 7 columns: Center, right and left camera. Sterring, throttle, brake and speed are other four columns. 

https://github.com/parthasen/SDC/blob/P3/index.png

Steering angle's boundary is from -1 to +1.

#### 3. Preprocessing and Approach
1.Steering angle =0 is excluded assuming the straight drive

2.
img=randomise_image_brightness(img)

img=togray(img)

trans_image()

Image is preprocessed like normalization ( using lamda function lambda x: x/255.-0.5) ,translation, grayed image and change of brightness (HSV)

I have selected learning rate of 0.0001 rather than the default adam optimizer rate of 0.001 to reduce loss.  Batch size considered here is 64 smaller than 128 usual size. I have tested with both the sizes but I found better training result by taking smaller size. I decided to fix 50 epochs after testing for 30 and 40.

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


#### 4. Model
1."elu" is used in this model as it is easier to use in comparison to "ReLU"

2.Strided convolutions are used at first three convolutional layers with a 2×2 stride and a 5×5 kernel

3.Non-strided convolution layers are used in the last two layers with a 3×3 kernel size. 

##### Visualization of model as https://github.com/parthasen/SDC/blob/P3/model-visualization.png taken of NVIDIA paper. 

model = Sequential()

model.add(Lambda(resize, input_shape=shape))

model.add(Lambda(lambda x: x/255.-0.5))

model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))

model.add(SpatialDropout2D(0.2))

model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))

model.add(SpatialDropout2D(0.2))

model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2,2), activation="elu"))

model.add(SpatialDropout2D(0.2))

model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))

model.add(SpatialDropout2D(0.2))

model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))

model.add(SpatialDropout2D(0.2))

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(100, activation="elu"))

model.add(Dense(50, activation="elu"))

model.add(Dense(10, activation="elu"))

model.add(Dropout(0.5))

model.add(Dense(1))

##### Layer (type)      ========================>              Output Shape                               
====================================================================================================
lambda_1 (Lambda)       =========================>           (None, 66, 200, 3)                    
____________________________________________________________________________________________________
lambda_2 (Lambda)       =========================>           (None, 66, 200, 3)                            
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D) =================>           (None, 33, 100, 24)                          
____________________________________________________________________________________________________
spatialdropout2d_1 (SpatialDropo ================>           (None, 33, 100, 24)                  
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  ================>           (None, 17, 50, 36)        
____________________________________________________________________________________________________
spatialdropout2d_2 (SpatialDropo ================>           (None, 17, 50, 36)                     
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  ================>           (None, 7, 23, 48)          
____________________________________________________________________________________________________
spatialdropout2d_3 (SpatialDropo ================>           (None, 7, 23, 48)             
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  ================>           (None, 5, 21, 64)               
____________________________________________________________________________________________________
spatialdropout2d_4 (SpatialDropo ================>           (None, 5, 21, 64)             
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  ================>           (None, 3, 19, 64)                 
____________________________________________________________________________________________________
spatialdropout2d_5 (SpatialDropo ================>           (None, 3, 19, 64)             
____________________________________________________________________________________________________
flatten_1 (Flatten) =============================>           (None, 3648)             
____________________________________________________________________________________________________
dropout_1 (Dropout) =============================>           (None, 3648)                 
____________________________________________________________________________________________________
dense_1 (Dense)     =============================>           (None, 100)                        
____________________________________________________________________________________________________
dense_2 (Dense)   ===============================>           (None, 50)                         
____________________________________________________________________________________________________
dense_3 (Dense)      ============================>           (None, 10)                 
____________________________________________________________________________________________________
dropout_2 (Dropout)     =========================>           (None, 10)                      
____________________________________________________________________________________________________
dense_4 (Dense)         =========================>           (None, 1)            


### 5.Result 

I found final training loss of 0.0348 and validation loss of 0.0301. This model drives the car well on both tracks (best performance at smallest resolution and lowest graphics), without ever crashing or venturing into dangerous areas. I used Xeon CPU only and observed around 40s is taken for each epoch. 

#### Simulation

https://www.youtube.com/watch?v=yU9Hs0lAvh0


<iframe width="560" height="315" src="https://www.youtube.com/embed/yU9Hs0lAvh0" frameborder="0" allowfullscreen></iframe>

https://www.youtube.com/watch?v=TtLeXqulQ3s
