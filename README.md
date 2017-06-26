# SDC : Extended Kalman Filter
 ![Calibration result](https://github.com/parthasen/SDC/blob/P6/Screenshot%20from%202017-06-26%2016-59-31.png)

1. Cloned the repo 
https://github.com/udacity/CarND-Extended-Kalman-Filter-Project
2. Simulator which can be downloaded and v.1.45 is used
https://github.com/udacity/self-driving-car-sim/releases
3. Installed uWebSocketIO from the repo (step 1) 
install-ubuntu.sh
4. Codes added at TODO section of  these only src/FusionEKF.cpp, kalman_filter.cpp, tools.cpp
5. Basic Build is done and no IDE used
    mkdir build
    cd build
    cmake ..
    make
    ./ExtendedKF

6. Code Style: 
Please (do your best to) stick to Google's C++ style guide. https://google.github.io/styleguide/cppguide.html

7. visualization and data generation utilities: 
https://github.com/udacity/CarND-Mercedes-SF-Utilities/tree/master/python

8. 
      INPUT: values provided by the simulator to the c++ program 
     ["sensor_measurement"] => the measurment that the simulator observed (either lidar or radar)


     OUTPUT: values provided by the c++ program to the simulator 
  ["estimate_x"] <= kalman filter estimated position x
  ["estimate_y"] <= kalman filter estimated position y
  ["rmse_x"]
  ["rmse_y"]
  ["rmse_vx"]
  ["rmse_vy"]
   ---
9. Project Rubric
https://review.udacity.com/#!/rubrics/748/view
px, py, vx, and vy RMSE should be less than or equal to the values [.11, .11, 0.52, 0.52]. 
