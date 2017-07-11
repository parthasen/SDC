#include <uWS/uWS.h>
#include <iostream>
#include "json.hpp"
#include "PID.h"
#include <math.h>
#include <vector>

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_last_of("]");
  if (found_null != std::string::npos) {
    return "";
  }
  else if (b1 != std::string::npos && b2 != std::string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

// Added for "twiddling" the PID coefficients



          // Twiddle
          // Implementation of the following python twiddle algorithm
          // as a state-machine.
          // Twiddle can be executed by running the PID program like "./pid true"
          //
          // python twiddle loop
          // while sum(dp) > tol:
          //   for i in range(len(p)):
          //     p[i] += dp[i]
          //     _, _, err = run(robot, p)
          //     if err < best_err:
          //         best_err = err
          //         dp[i] *= 1.1
          //     else:
          //         p[i] -= 2.0 * dp[i]
          //         _, _, err = run(robot, p)
          //         if err < best_err:
          //             best_err = err
          //             dp[i] *= 1.1
          //        else:
          //             p[i] += dp[i]
          //             dp[i] *= 0.9


double twiddle_best_error_ = 1000000;
bool twiddle_state_ = 0;
int twiddle_idx = 0;
int twiddle_iterations_ = 0;
std::vector<double> p = {0.1, 0.001, 3.0};
std::vector<double> dp = {0.5, 0.001,4};

void twiddle(PID &pid_control) {
  std::cout << "State: " << twiddle_state_ << std::endl;
  std::cout << "PID Error: " << pid_control.TotalError() << ", Best Error: " << twiddle_best_error_ << std::endl;
  if (twiddle_state_ == 0) {
    twiddle_best_error_ = pid_control.TotalError();
    p[twiddle_idx] += dp[twiddle_idx];
    twiddle_state_ = 1;
  } else if (twiddle_state_ == 1) {
    if (pid_control.TotalError() < twiddle_best_error_) {
      twiddle_best_error_ = pid_control.TotalError();
      dp[twiddle_idx] *= 1.1;
      twiddle_idx = (twiddle_idx + 1) % 3; //rotate over the 3 vector indices
      p[twiddle_idx] += dp[twiddle_idx];
      twiddle_state_ = 1;
    } else {
      p[twiddle_idx] -= 2 * dp[twiddle_idx];
      if (p[twiddle_idx] < 0) {
        p[twiddle_idx] = 0;
        twiddle_idx = (twiddle_idx + 1) % 3;
      }
      twiddle_state_ = 2;
    }
  } else { //twiddle_state_ = 2
    if (pid_control.TotalError() < twiddle_best_error_) {
      twiddle_best_error_ = pid_control.TotalError();
      dp[twiddle_idx] *= 1.1;
      twiddle_idx = (twiddle_idx + 1) % 3;
      p[twiddle_idx] += dp[twiddle_idx];
      twiddle_state_ = 1;
    } else {
      p[twiddle_idx] += dp[twiddle_idx];
      dp[twiddle_idx] *= 0.9;
      twiddle_idx = (twiddle_idx + 1) % 3;
      p[twiddle_idx] += dp[twiddle_idx];
      twiddle_state_ = 1;
      //pid.Init(p[0], p[1], p[2]);
    }
  }

  pid_control.Init(p[0], p[1], p[2]);
}
// End Twiddle


int main()
{
  std::cout << "Twiddling PID of steering and speed" << std::endl;
 
  uWS::Hub h;

  PID pid;
  PID throttlePid;
  // TODO: Initialize the pid variable.
  //Kp = 0.15;//0.08 //0.1//0.15//
  //Ki = 0.001;//0.003 //0.005//.0001
  //Kd = 1.5;//5 //4
  double init_speed =15; // mph
  //(0.2, 0.004, 3.0);
  //(0.2, 0.001, 3.0);
  pid.Init(0.2, 0.002,2.5); // best params so far found with twiddle on continuous loop
  throttlePid.Init(0.15, 0.004,5);
  h.onMessage([&pid, &throttlePid, init_speed](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2')
    {
      auto s = hasData(std::string(data));
      if (s != "") {
        auto j = json::parse(s);
        std::string event = j[0].get<std::string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          double cte = std::stod(j[1]["cte"].get<std::string>());
          double speed = std::stod(j[1]["speed"].get<std::string>());
          //double angle = std::stod(j[1]["steering_angle"].get<std::string>());

          double steer_value;
          double throttle;
          double speed_error;
	  speed_error =speed-init_speed;
          

          /*
          * Calcuate steering value here, remember the steering value is
          * [-1, 1].
          * NOTE: Another PID controller is used to set the speed.
          */
          pid.UpdateError(cte);
          steer_value = pid.TotalError();
    	  steer_value = std::max(-1.0,std::min(1.0, steer_value));////-1.1,1.1
          throttlePid.UpdateError(speed_error);
          throttle = throttlePid.TotalError();
          //throttle = std::max(-1.1, std::min(1.1, throttle));
          
          std::cout << "CTE: " << cte << " Steering Value: " << steer_value << " Speed: " << throttle << std::endl;
          twiddle_iterations_++;
  
          // Let it start running a bit first and also reset if car crashes //avoiding crash
          if ((twiddle_iterations_ > 5000) || ((speed<init_speed*0.75) && twiddle_iterations_ > 800)) {
             if ((speed<init_speed*0.75) && twiddle_iterations_ > 800) {
                twiddle_best_error_ = 1000000;
                std::string msg = "42[\"reset\", {}]";
                std::cout << msg << std::endl;
                ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
              }

              twiddle(pid);
              twiddle(throttlePid);
              std::cout << "P VECTOR: " << p[0] << "\t" << p[1] << "\t" << p[2] << std::endl;
              twiddle_iterations_ = 0;
            }
	    else {	
            json msgJson;
            msgJson["steering_angle"] = steer_value;
            msgJson["throttle"] = throttle;
            auto msg = "42[\"steer\"," + msgJson.dump() + "]";
            ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
		}
          } 
  
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1)
    {
      res->end(s.data(), s.length());
    }
    else
    {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port))
  {
    std::cout << "Listening to port " << port << std::endl;
  }
  else
  {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
