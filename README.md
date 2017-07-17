# SDC
self driving car related everything

# CarND-Controls-MPC
Self-Driving Car Engineer Nanodegree Program

---
## Steps

1. I was getting error "# error "don't have header file for stddef" when I was using sudo apt-get install coinor-libipopt-dev. So I downloaded Ipopt 3.12.8 from https://www.coin-or.org/download/source/Ipopt/Ipopt-3.12.8.zip and saved as a sibling to install_ipopt.sh. Then installed using "sudo bash install_ipopt.sh Ipopt-3.12.8"
2. CppAD is installed using `sudo apt-get install cppad`
3. uWebSockets already installed and Eigen is saved in src folder.
4. IDE is not used so I followed Basic Build Instructions
    * Make a build directory: `mkdir build && cd build`
    * Compile: `cmake .. && make`
    * Run it: `./mpc`.
5.dt less than 0.1 showed bad control in low and high speed. [8,0.08,90],[6,0.05,70].
6. N=12 and dt=0.12 resulted good control. For this timestep length and duration speed tested at 70, 80. And highest speed crossed 80 many times. So speed was initialized to 90 to test. But control was not good.
7. Lastly N=8, dt=0.12 and ref_v = 85 was insitialized.

## Model
#### Steps:

1. Set N and dt.
2. Fit the polynomial to the waypoints.
3. Calculate initial cross track error and orientation error values.
4. Define the components of the cost function (state, actuators, etc). You may use the methods previously discussed or make up something, up to you!
5. Define the model constraints. These are the state update equations defined in the Vehicle Models module.

#### State and actuator
Position (x,y), heading (ψ) and velocity (v) form the vehicle state vector. State is [x, y, psi, v] and actuators is [delta, a]
#### update 
Once again, the model we’ve developed:

x​t+1​​=x​t​​+v​t​​∗cos(ψ​t​​)∗dt

y​t+1​​=y​t​​+v​t​​∗sin(ψ​t​​)∗dt

ψ​t+1​​=ψ​t​​+​L​f​​​​v​t​​​​∗δ∗dt

v​t+1​​=v​t​​+a​t​​∗dt

cte​t+1​​=f(x​t​​)−y​t​​+(v​t​​∗sin(eψ​t​​)∗dt)

eψ​t+1​​=ψ​t​​−ψdes​t​​+(​L​f​​​​v​t​​​​∗δ​t​​∗dt)

We’ve added a variable to our state called L​f​​ which measures the distance between the front of the vehicle and its center of gravity. The larger the vehicle , the slower the turn rate. if δ is positive we rotate counter-clockwise, or turn left. In the simulator however, a positive value implies a right turn and a negative value implies a left turn.
Two possible ways to get around this are:
Change the update equation to ψ​t+1​​=ψ​t​​−​L​f​​​​v​t​​​​∗δ​t​​∗dt
Leave the update equation as is and multiply the steering value by -1 before sending it back to the server.

#### Errors
Cross track error (cte) and ψ error (eψ) were used to build the cost function for the MPC. We can capture how the errors we are interested in change over time by deriving our kinematic model around these errors as our new state vector.

The new state is [x,y,ψ,v,cte,eψ].

