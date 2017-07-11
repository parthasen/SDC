# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

---

# Reflection

https://www.youtube.com/watch?v=k0-r4fNGXlU&feature=youtu.be

[![Video Output](https://i.ytimg.com/vi/k0-r4fNGXlU/2.jpg?time=1499790278979)](https://www.youtube.com/watch?v=k0-r4fNGXlU&feature=youtu.be)


1. High initial speed lead to quick crash of the car. I started from 80 but reduced to 15. 
2. First I tried without twiddling but control was not good. So iplemented twidle for better control. Twidle was first implemented with steering angle only but ended project twiddling both steering and speed.
3. For steering PID many hyperparameters like (0.2, 0.004, 3.0), (0.2, 0.001, 3.0), (0.15, 0.004,2.2) were tested along with different speed but lastly low initial speed was fixed. 
4. Hyperparameter for speed PID was teseted from a range (0.1, 0.001,2) to (0.45, 0.005,5). 
5. Steer value range used [-1.1 to 1.1]
6. Fixed PID hyperparameter for steering (0.2, 0.002,2.5) and for speed (0.15, 0.004,5)
7. Twiddle best error is initialized with 1000000
8. Minimum twiddle iteration is 800 after which car will be stopped to avoid crash
 



