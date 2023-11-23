#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */
   VectorXd rmse(4);
   rmse << 0,0,0,0;

   if(estimations.size()==0 || estimations.size()!=ground_truth.size())
      return rmse;
   for (int i=0; i < estimations.size(); ++i) {
      rmse.array() += (estimations[i]-ground_truth[i]).array()* (estimations[i]-ground_truth[i]).array();
   }

   rmse = (rmse/estimations.size()).array().sqrt();
   return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * TODO:
   * Calculate a Jacobian here.
   */
  MatrixXd Hj(3,4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // check division by zero
  if(px==0. && py == 0.)
    return Hj;
  // compute the Jacobian matrix
  float dist = px*px+py*py;
  Hj << px/sqrt(dist), py/sqrt(dist), 0, 0,
        -py/dist, px/dist, 0, 0,
        py*(vx*py-vy*px)/pow(dist, 3/(double)2), px*(vy*px-vx*py)/pow(dist, 3/(double)2), px/sqrt(dist), py/sqrt(dist);

  return Hj;
}
