#include "kalman_filter.h"
#include <iostream> 

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
   * TODO: predict the state
   */
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * TODO: update the state by using Kalman Filter equations
   */
  VectorXd y = z - H_ * x_;
  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K = P_ * H_.transpose() * S.inverse();
  x_ = x_ + K * y;
  // Identity matrix
  long x_size = x_.size();
  Eigen::MatrixXd I_ = Eigen::MatrixXd::Identity(x_size, x_size);
  P_ = (I_ - K*H_)*P_;
}

void change_angle_range(double& angle)
{
  if(angle > M_PI)
    angle -= 2 * M_PI;
  else if(angle < -M_PI)
    angle += 2 * M_PI;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * TODO: update the state by using Extended Kalman Filter equations
   */
  VectorXd z_pred = VectorXd(3);
  z_pred(0) = sqrt(x_[0]*x_[0]+x_[1]*x_[1]);
  z_pred(1) = atan2(x_[1], x_[0]);
  //avoid 0
  if(fabs(z_pred(0)) < 1e-4)
  {
    z_pred(2) = 0;
  }
  else{
    z_pred(2) = (x_[0]*x_[2]+x_[1]*x_[3])/z_pred(0);
  }
  
  VectorXd y = z-z_pred;
  //  the resulting angle phi in the y vector should be adjusted so that it is between -pi and pi.
  while(!(y[1]>-M_PI && y[1]<M_PI))
  {
    change_angle_range(y(1));
  }
  MatrixXd S = H_*P_*H_.transpose() + R_;
  MatrixXd K = P_*H_.transpose()*S.inverse();
  x_ = x_+K*y;
  // Identity matrix
  long x_size = x_.size();
  Eigen::MatrixXd I_ = Eigen::MatrixXd::Identity(x_size, x_size);
  P_ = (I_ - K*H_)*P_;
}
